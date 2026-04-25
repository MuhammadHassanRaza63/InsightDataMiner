import os
import io
import base64
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

from django import forms
from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.http import JsonResponse, FileResponse
from django.utils import timezone
from .models import UserModuleUsage, UserReport

class ModernSignupForm(forms.ModelForm):
    username = forms.CharField(max_length=150, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'}))
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email Address'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Strict Password'}))
    confirm_password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Confirm Password'}))

    class Meta:
        model = User
        fields = ['username', 'email']

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")
        if password and confirm_password:
            if password != confirm_password: self.add_error('confirm_password', "Passwords do not match!")
            if len(password) < 8: self.add_error('password', "Must be at least 8 characters.")
            if not re.search(r'[A-Z]', password): self.add_error('password', "Must contain an Uppercase letter.")
            if not re.search(r'[a-z]', password): self.add_error('password', "Must contain a Lowercase letter.")
            if not re.search(r'\d', password): self.add_error('password', "Must contain a Number.")
            if not re.search(r'[@$!%*?&#]', password): self.add_error('password', "Must contain a Special Character (@$!%*?&#).")
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password"])
        if commit: user.save()
        return user

def home(request): return render(request, 'upload.html')
def signup_view(request):
    if request.method == 'POST':
        form = ModernSignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else: form = ModernSignupForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else: form = AuthenticationForm()
    for field in form.fields.values(): field.widget.attrs['class'] = 'form-control'
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('home')


# ==========================================
# MODULE 1 (PREPROCESSING)
# ==========================================
def module1_workspace(request):
    context = {}
    fs = FileSystemStorage()
    theme = request.COOKIES.get('idm_theme', 'dark')
    tick_color = '#0f172a' if theme in ['light', 'hybrid'] else '#f8fafc'
    plt.rcParams.update({'text.color': tick_color, 'axes.labelcolor': tick_color, 'xtick.color': tick_color, 'ytick.color': tick_color})

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'load_sample':
            try:
                df_sample = sns.load_dataset('titanic')
            except:
                df_sample = pd.DataFrame({'Age': [22, 38, 26, 35, np.nan], 'Fare': [7.25, 71.28, 7.92, 53.1, 8.45], 'Survived': [0, 1, 1, 1, 0], 'Class': ['Third', 'First', 'Third', 'First', 'Third']})
            filename = 'titanic_sample.csv'
            file_path = os.path.join(settings.MEDIA_ROOT, filename)
            df_sample.to_csv(file_path, index=False)
            request.session['active_dataset'] = file_path
            context['filename'] = filename
            context['success_msg'] = "Sample Dataset loaded successfully!"

        elif action == 'upload_file' and request.FILES.get('dataset'):
            uploaded_file = request.FILES['dataset']
            if not uploaded_file.name.endswith(('.csv', '.xlsx', '.xls', '.json')):
                context['error'] = "Unsupported Format! Please provide a valid file."
            else:
                filename = fs.save(uploaded_file.name, uploaded_file)
                request.session['active_dataset'] = fs.path(filename)
                context['filename'] = filename
                context['success_msg'] = "Dataset successfully ingested."

        elif action == 'execute_pipeline':
            filename = request.POST.get('filename')
            file_path = os.path.join(settings.MEDIA_ROOT, filename)
            try:
                df = pd.read_csv(file_path) if filename.endswith('.csv') else pd.read_json(file_path) if filename.endswith('.json') else pd.read_excel(file_path)
                orig_rows, orig_cols = df.shape
                orig_missing, orig_memory = int(df.isnull().sum().sum()), df.memory_usage(deep=True).sum() / 1024 
                orig_cats = len(df.select_dtypes(include=['object', 'category']).columns)
                orig_health = round(100 * (1 - (orig_missing / (orig_rows * orig_cols))), 1) if (orig_rows * orig_cols) > 0 else 0
                outliers_fixed = 0

                remove_input = request.POST.get('remove_cols', '').strip()
                if remove_input: df.drop(columns=[c.strip() for c in remove_input.split(',') if c.strip() in df.columns], inplace=True)
                
                if request.POST.get('rename_cols', '').strip():
                    try: df.rename(columns={k.strip(): v.strip() for k, v in dict(i.split(':') for i in request.POST.get('rename_cols').split(',') if ':' in i).items()}, inplace=True)
                    except: pass

                if request.POST.get('do_clean'):
                    df.drop(columns=[c for c in df.columns if df[c].isnull().sum()/len(df) > 0.5], inplace=True)
                    for col in df.columns:
                        if df[col].isnull().sum() > 0:
                            if df[col].dtype in ['int64', 'float64']: df[col] = df[col].fillna(df[col].median())
                            elif not df[col].mode().empty: df[col] = df[col].fillna(df[col].mode()[0])

                plot_data = []
                if request.POST.get('do_outliers'):
                    for col in df.select_dtypes(include=[np.number]).columns[:4]:
                        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                        outliers_fixed += len(df[(df[col] < lower) | (df[col] > upper)])
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                        sns.boxplot(x=df[col], ax=ax1, color='#ef4444')
                        df[col] = np.clip(df[col], lower, upper)
                        sns.boxplot(x=df[col], ax=ax2, color='#10b981')
                        buf = io.BytesIO()
                        plt.tight_layout()
                        plt.savefig(buf, format='png', transparent=True)
                        plot_data.append({'column': col, 'chart': base64.b64encode(buf.getvalue()).decode('utf-8')})
                        plt.close(fig)

                if request.POST.get('do_encode'):
                    for col in df.select_dtypes(include=['object', 'category']).columns:
                        df[col] = df[col].astype('category').cat.codes

                if request.POST.get('do_scale'):
                    for col in df.select_dtypes(include=[np.number]).columns:
                        if df[col].std() != 0: df[col] = (df[col] - df[col].mean()) / df[col].std()

                final_rows, final_cols = df.shape
                final_missing, final_memory = int(df.isnull().sum().sum()), df.memory_usage(deep=True).sum() / 1024
                final_cats = len(df.select_dtypes(include=['object', 'category']).columns)
                final_health = 100.0 if final_missing == 0 else round(100 * (1 - (final_missing / (final_rows * final_cols))), 1)

                df.to_csv(file_path, index=False)
                context.update({
                    'show_results': True, 'plot_data': plot_data, 'filename': filename, 'success_msg': "Pipeline Executed Successfully!",
                    'comparison': {'before': {'rows': orig_rows, 'cols': orig_cols, 'missing': orig_missing, 'memory': f"{orig_memory:.1f} KB", 'cats': orig_cats, 'health': f"{orig_health}%"},
                                   'after': {'rows': final_rows, 'cols': final_cols, 'missing': final_missing, 'memory': f"{final_memory:.1f} KB", 'cats': final_cats, 'health': f"{final_health}%", 'outliers_fixed': outliers_fixed}}
                })

                # Track module usage
                if request.user.is_authenticated:
                    UserModuleUsage.objects.update_or_create(
                        user=request.user,
                        module='module1',
                        defaults={
                            'dataset_name': filename,
                            'session_data': {
                                'original_stats': context['comparison']['before'],
                                'processed_stats': context['comparison']['after'],
                                'operations': {
                                    'removed_cols': request.POST.get('remove_cols', ''),
                                    'renamed_cols': request.POST.get('rename_cols', ''),
                                    'cleaned': bool(request.POST.get('do_clean')),
                                    'outliers_fixed': bool(request.POST.get('do_outliers')),
                                    'encoded': bool(request.POST.get('do_encode')),
                                    'scaled': bool(request.POST.get('do_scale'))
                                }
                            }
                        }
                    )
            except Exception as e: context['error'] = f"Processing Error: {str(e)}"

    if 'filename' in context:
        try:
            df_pv = pd.read_csv(os.path.join(settings.MEDIA_ROOT, context['filename']))
            context.update({'total_rows': df_pv.shape[0], 'total_cols': df_pv.shape[1], 'total_missing': int(df_pv.isnull().sum().sum()), 'data_html': df_pv.head(5).to_html(classes='table table-hover mb-0' + (' table-dark' if theme == 'dark' else ''), index=False, border=0)})
        except: pass
    return render(request, 'module1.html', context)


# ==========================================
# MODULE 2 (VISUAL ANALYTICS)
# ==========================================
def module2_visual_analytics(request):
    context = {'module': 'module2'}
    theme = request.COOKIES.get('idm_theme', 'dark')
    plot_template = 'plotly_white' if theme in ['light', 'hybrid'] else 'plotly_dark'
    font_color = '#475569' if theme in ['light', 'hybrid'] else '#94a3b8'
    if 'pinned_charts' not in request.session: request.session['pinned_charts'] = []

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'load_sample':
            try: df = sns.load_dataset('titanic')
            except: df = pd.DataFrame({'Age': [22, 38, 26, 35], 'Fare': [7.25, 71.28, 7.92, 53.1], 'Survived': [0, 1, 1, 1], 'Class': ['Third', 'First', 'Third', 'First']})
            file_path = os.path.join(settings.MEDIA_ROOT, 'titanic_sample.csv')
            df.to_csv(file_path, index=False)
            request.session['active_dataset'] = file_path
            request.session['pinned_charts'] = []
            context['success_msg'] = "Sample Dataset loaded!"
            
        elif 'new_dataset' in request.FILES:
            fs = FileSystemStorage()
            filename = fs.save(request.FILES['new_dataset'].name, request.FILES['new_dataset'])
            request.session['active_dataset'] = fs.path(filename)
            request.session['pinned_charts'] = []
            context['success_msg'] = "Dataset Loaded!"

    filepath = request.session.get('active_dataset', None)
    if not filepath or not os.path.exists(filepath):
        context['standby'] = True
        return render(request, 'module2.html', context)
        
    try:
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_json(filepath) if filepath.endswith('.json') else pd.read_excel(filepath)
        num_cols, cat_cols = df.select_dtypes(include=['number']).columns.tolist(), df.select_dtypes(exclude=['number']).columns.tolist()
        context.update({'numeric_cols': num_cols, 'categorical_cols': cat_cols, 'all_cols': df.columns.tolist(), 'dataset_name': os.path.basename(filepath), 'total_rows': len(df)})
        context['current_config'] = {'type': '', 'x': '', 'y': '', 'color': ''}

        def compute_best_pair():
            suggestion = {'type': '', 'x': '', 'y': '', 'color': None, 'description': 'No recommended pair available.'}
            if len(num_cols) > 1:
                corr = df[num_cols].corr().abs()
                best = ('', '', -1)
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        value = corr.iloc[i, j]
                        if value > best[2]:
                            best = (num_cols[i], num_cols[j], value)
                suggestion = {
                    'type': 'scatter',
                    'x': best[0],
                    'y': best[1],
                    'color': None,
                    'description': f"Best correlated numeric pair: {best[0]} vs {best[1]} (r={best[2]:.2f})"
                }
            elif num_cols and cat_cols:
                suggestion = {
                    'type': 'bar',
                    'x': cat_cols[0],
                    'y': num_cols[0],
                    'color': cat_cols[0],
                    'description': f"Strong categorical grouping available: {cat_cols[0]} vs {num_cols[0]}"
                }
            elif num_cols:
                suggestion = {
                    'type': 'histogram',
                    'x': num_cols[0],
                    'y': None,
                    'color': None,
                    'description': f"Plot distribution for {num_cols[0]} to explore spread and outliers."
                }
            elif cat_cols:
                suggestion = {
                    'type': 'count_bar',
                    'x': cat_cols[0],
                    'y': None,
                    'color': None,
                    'description': f"Count the observations across {cat_cols[0]} categories."
                }
            return suggestion

        def summarise_numeric_columns():
            stats = []
            for col in num_cols[:4]:
                try:
                    stats.append({
                        'column': col,
                        'count': int(df[col].count()),
                        'sum': round(df[col].sum(), 2),
                        'avg': round(df[col].mean(), 2),
                        'min': round(df[col].min(), 2),
                        'max': round(df[col].max(), 2)
                    })
                except Exception:
                    continue
            return stats

        context['best_pair_suggestion'] = compute_best_pair()
        context['numeric_stats'] = summarise_numeric_columns()

        def get_plotly_fig(c_type, x, y, color):
            fig = None
            try:
                if c_type == 'scatter' and x and y: fig = px.scatter(df, x=x, y=y, color=color, template=plot_template)
                elif c_type == 'bar' and x and y: fig = px.bar(df, x=x, y=y, color=color, template=plot_template, barmode='group')
                elif c_type == 'line' and x and y: fig = px.line(df, x=x, y=y, color=color, template=plot_template)
                elif c_type == 'histogram' and x: fig = px.histogram(df, x=x, color=color, template=plot_template)
                elif c_type == 'count_bar' and x: fig = px.histogram(df, x=x, color=color, template=plot_template)
                elif c_type == 'box' and y: fig = px.box(df, x=x, y=y, color=color, template=plot_template)
                elif c_type == 'pie' and x and y: fig = px.pie(df, names=x, values=y, template=plot_template)
                elif c_type == 'heatmap': fig = px.imshow(df[num_cols].corr(), text_auto=True, aspect="auto", template=plot_template)
                if fig: fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=font_color), margin=dict(l=20, r=20, t=50, b=20))
            except: pass
            return fig

        plot_config = {'displaylogo': False}
        if not request.user.is_authenticated: plot_config['modeBarButtonsToRemove'] = ['toImage']; context['guest_warning'] = True

        if request.method == 'POST':
            action = request.POST.get('action')
            if action == 'clear_dashboard':
                request.session['pinned_charts'] = []
                context['success_msg'] = "Dashboard cleared!"
            elif action == 'auto_dashboard':
                missing_total = int(df.isnull().sum().sum())
                missing_pct = round((missing_total / df.size) * 100, 1) if df.size else 0
                numeric_count = len(num_cols)
                categorical_count = len(cat_cols)

                column_summaries = []
                for col in num_cols[:3]:
                    try:
                        column_summaries.append({
                            'name': col,
                            'mean': round(df[col].mean(), 2),
                            'median': round(df[col].median(), 2),
                            'std': round(df[col].std(), 2),
                            'min': round(df[col].min(), 2),
                            'max': round(df[col].max(), 2)
                        })
                    except Exception:
                        pass

                top_category = None
                if cat_cols:
                    counts = df[cat_cols[0]].value_counts()
                    if not counts.empty:
                        top_category = f"{counts.idxmax()} ({int(counts.max())} records)"

                auto_insights = []
                if missing_total > 0:
                    auto_insights.append(f"{missing_total} missing values found ({missing_pct}%).")
                else:
                    auto_insights.append("No missing values detected; data quality looks strong.")
                if numeric_count > 1:
                    corr = df[num_cols].corr().abs().unstack().sort_values(ascending=False)
                    strong_corr = corr[(corr < 1.0) & (corr > 0.75)]
                    if not strong_corr.empty:
                        pair = strong_corr.index[0]
                        auto_insights.append(f"Strong relationship between {pair[0]} and {pair[1]} ({strong_corr.iloc[0]:.2f}).")
                    else:
                        auto_insights.append("No overly strong numeric correlations detected; features are diverse.")
                if top_category:
                    auto_insights.append(f"Top category in {cat_cols[0]} is {top_category}.")
                if num_cols:
                    auto_insights.append(f"Average {num_cols[0]} is {df[num_cols[0]].mean():.2f}.")

                chart_list = []
                if num_cols:
                    hist_chart = get_plotly_fig('histogram', num_cols[0], None, None)
                    if hist_chart: chart_list.append(hist_chart)
                    box_chart = get_plotly_fig('box', None, num_cols[0], None)
                    if box_chart: chart_list.append(box_chart)
                    if len(num_cols) > 1:
                        scatter_chart = get_plotly_fig('scatter', num_cols[0], num_cols[1], None)
                        if scatter_chart: chart_list.append(scatter_chart)
                    corr_chart = get_plotly_fig('heatmap', None, None, None)
                    if corr_chart: chart_list.append(corr_chart)
                if cat_cols:
                    cat_chart = get_plotly_fig('count_bar', cat_cols[0], None, None)
                    if cat_chart: chart_list.append(cat_chart)

                context['auto_summary'] = {
                    'rows': len(df),
                    'missing': missing_total,
                    'missing_pct': missing_pct,
                    'numeric_cols': numeric_count,
                    'categorical_cols': categorical_count,
                    'features': len(df.columns),
                    'health': f"{max(0, min(100, 100 - missing_pct))}%",
                    'avg_col': round(df[num_cols[0]].mean(), 2) if num_cols else None,
                    'top_category': top_category
                }
                context['auto_insights'] = auto_insights
                context['column_summaries'] = column_summaries
                context['auto_dashboard_charts'] = [chart.to_html(full_html=False, config=plot_config) for chart in chart_list[:4]]
                context['success_msg'] = "Dashboard Generated!"
            elif action in ['generate_chart', 'pin_chart']:
                c_type, x_col, y_col, c_col = request.POST.get('chart_type'), request.POST.get('x_axis'), request.POST.get('y_axis'), request.POST.get('color_col')
                if c_col == 'none': c_col = None
                current_config = {'type': c_type or '', 'x': x_col or '', 'y': y_col or '', 'color': c_col}
                settings_dict = current_config.copy()
                if action == 'pin_chart':
                    request.session['pinned_charts'] = request.session.get('pinned_charts', []) + [settings_dict]
                    context['success_msg'] = "Chart Pinned!"
                fig = get_plotly_fig(c_type, x_col, y_col, c_col)
                if fig:
                    context.update({'chart_html': fig.to_html(full_html=False, config=plot_config), 'current_config': current_config})
                else:
                    context.update({'current_config': current_config})
                    context['error'] = "Invalid chart parameters."

        # Track module usage when user interacts with module2
        if request.user.is_authenticated and request.method == 'POST':
            UserModuleUsage.objects.update_or_create(
                user=request.user,
                module='module2',
                defaults={
                    'dataset_name': os.path.basename(filepath) if filepath else None,
                    'session_data': {
                        'pinned_charts': request.session.get('pinned_charts', []),
                        'total_charts': len(request.session.get('pinned_charts', [])),
                        'last_action': request.POST.get('action')
                    }
                }
            )

        context['dashboard_charts'] = [get_plotly_fig(p['type'], p['x'], p['y'], p['color']).update_layout(height=400).to_html(full_html=False, config=plot_config) for p in request.session.get('pinned_charts', []) if get_plotly_fig(p['type'], p['x'], p['y'], p['color'])]

    except Exception as e: context['error'] = f"Engine Error: {str(e)}"
    return render(request, 'module2.html', context)


# ==========================================
# MODULE 3 (AUTOML)
# ==========================================
def module3_automl(request):
    context = {'module': 'module3'}
    theme = request.COOKIES.get('idm_theme', 'dark')
    plot_template = 'plotly_white' if theme in ['light', 'hybrid'] else 'plotly_dark'
    font_color = '#475569' if theme in ['light', 'hybrid'] else '#94a3b8'

    if request.method == 'POST' and request.POST.get('action') == 'download_model':
        if os.path.exists(request.session.get('model_path', '')): return FileResponse(open(request.session['model_path'], 'rb'), as_attachment=True, filename='IDM_Trained_Model.pkl')

    if request.method == 'POST' and request.POST.get('action') == 'predict' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        try:
            with open(request.session['model_path'], 'rb') as f:
                md = pickle.load(f)

            in_data = {}
            for feature_name in md['features']:
                if feature_name in md['encoders']:
                    in_data[feature_name] = [md['encoders'][feature_name].transform([request.POST.get(feature_name)])[0]]
                else:
                    in_data[feature_name] = [float(request.POST.get(feature_name) or 0)]

            df_in = pd.DataFrame(in_data)
            if md.get('numeric_features'):
                df_in[md['numeric_features']] = md['scaler'].transform(df_in[md['numeric_features']])

            pred = md['model'].predict(df_in)[0]
            if md.get('target_encoder'):
                pred = md['target_encoder'].inverse_transform([int(pred)])[0]
            return JsonResponse({'status': 'success', 'prediction': str(round(pred, 2) if isinstance(pred, float) else pred)})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    if request.method == 'POST':
        if request.POST.get('action') == 'load_sample':
            try: df = sns.load_dataset('titanic')
            except: df = pd.DataFrame({'Age': [22, 38, 26, 35], 'Fare': [7.25, 71.28, 7.92, 53.1], 'Survived': [0, 1, 1, 1], 'Class': ['Third', 'First', 'Third', 'First']})
            file_path = os.path.join(settings.MEDIA_ROOT, 'titanic_sample.csv')
            df.to_csv(file_path, index=False)
            request.session['active_dataset'] = file_path
            context['success_msg'] = "Sample Dataset loaded!"
        elif 'new_dataset' in request.FILES:
            fs = FileSystemStorage()
            filename = fs.save(request.FILES['new_dataset'].name, request.FILES['new_dataset'])
            request.session['active_dataset'] = fs.path(filename)
            context['success_msg'] = "Dataset Loaded!"
    
    filepath = request.session.get('active_dataset', None)
    if not filepath or not os.path.exists(filepath):
        context['standby'] = True
        return render(request, 'module3.html', context)
        
    try:
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_json(filepath) if filepath.endswith('.json') else pd.read_excel(filepath)
        context.update({'all_cols': df.columns.tolist(), 'dataset_name': os.path.basename(filepath), 'total_rows': len(df)})

        if request.method == 'POST' and request.POST.get('action') == 'run_automl':
            target_col, task_type = request.POST.get('target_col'), request.POST.get('task_type')
            if not target_col or not task_type:
                context['error'] = "Select both features."
            elif len(df.dropna()) < 10:
                context['error'] = "Not enough clean data."
            else:
                df_ml = df.dropna().copy()
                X, y = df_ml.drop(columns=[target_col]), df_ml[target_col]
                encoders = {col: LabelEncoder().fit(X[col].astype(str)) for col in X.select_dtypes(include=['object', 'category']).columns}
                for col, le in encoders.items():
                    X[col] = le.transform(X[col].astype(str))

                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                scaler = StandardScaler()
                if numeric_cols:
                    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

                target_enc = LabelEncoder().fit(y.astype(str)) if task_type == 'classification' else None
                if target_enc:
                    y = target_enc.transform(y.astype(str))
                elif task_type == 'regression' and y.dtype == object:
                    y = pd.to_numeric(y.astype(str), errors='coerce')

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                if task_type == 'classification':
                    models = {
                        'Logistic Regression': LogisticRegression(max_iter=2000, solver='liblinear'),
                        'Decision Tree': DecisionTreeClassifier(random_state=42),
                        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
                    }
                else:
                    models = {
                        'Linear Regression': LinearRegression(),
                        'Decision Tree': DecisionTreeRegressor(random_state=42),
                        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
                    }

                results, best_score, best_cv_score, best_model_inst, fit_suggestion = [], -float('inf'), -float('inf'), None, None
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        train_score = accuracy_score(y_train, model.predict(X_train))*100 if task_type == 'classification' else max(0, r2_score(y_train, model.predict(X_train))*100)
                        test_score = accuracy_score(y_test, model.predict(X_test))*100 if task_type == 'classification' else max(0, r2_score(y_test, model.predict(X_test))*100)
                        cv_metric = 'accuracy' if task_type == 'classification' else 'r2'
                        cv_scores = cross_val_score(model, X, y, cv=4, scoring=cv_metric)
                        cv_score = np.mean(cv_scores) * 100

                        fit_gap = train_score - test_score
                        cv_std = np.std(cv_scores) * 100
                        suggestion = []
                        if fit_gap > 15 and test_score > 50:
                            suggestion.append(f'Overfitting risk detected. Training score is {train_score:.1f}% vs test score {test_score:.1f}%.')
                            suggestion.append('Try simpler models, regularization, or more training data.')
                        elif test_score < (65 if task_type == 'classification' else 35):
                            suggestion.append(f'Underfitting risk detected. Test score is only {test_score:.1f}% on held-out data.')
                            suggestion.append('Try feature engineering, more data, or a stronger algorithm.')
                        elif cv_std > 12:
                            suggestion.append(f'High cross-validation variance ({cv_std:.1f}%) detected; the model may be unstable.')
                            suggestion.append('Consider regularization, more data, or feature selection.')
                        else:
                            suggestion.append('Model fit looks balanced across training, validation, and cross-validation.')
                        if cv_score < test_score - 10:
                            suggestion.append('Cross-validation suggests variance; consider stronger validation or data augmentation.')

                        status_value = 'balanced'
                        if fit_gap > 15 and test_score > 50:
                            status_value = 'overfit'
                        elif test_score < (65 if task_type == 'classification' else 35):
                            status_value = 'underfit'
                        elif cv_std > 12:
                            status_value = 'variance'
                        model_data = {
                            'model': name,
                            'score': round(test_score, 2),
                            'train_score': round(train_score, 2),
                            'cv_score': round(cv_score, 2),
                            'cv_std': round(cv_std, 2),
                            'fit_gap': round(fit_gap, 2),
                            'status': status_value,
                            'suggestion': ' '.join(suggestion)
                        }
                        results.append(model_data)

                        if test_score > best_score or (test_score == best_score and cv_score > best_cv_score):
                            best_score = test_score
                            best_cv_score = cv_score
                            best_model_inst = model
                            fit_suggestion = model_data['suggestion']
                    except Exception:
                        continue

                results = sorted(results, key=lambda x: (x['score'], x['cv_score']), reverse=True)
                if not results:
                    context['error'] = 'No valid models were trained. Please check your data and target selection.'
                else:
                    pkl_path = os.path.join(settings.MEDIA_ROOT, 'best_trained_model.pkl')
                    with open(pkl_path, 'wb') as f:
                        pickle.dump({
                            'model': best_model_inst,
                            'scaler': scaler,
                            'encoders': encoders,
                            'target_encoder': target_enc,
                            'features': X.columns.tolist(),
                            'numeric_features': numeric_cols
                        }, f)
                    request.session['model_path'] = pkl_path

                    plot_config = {'displaylogo': False}
                    fig_comp = px.bar(pd.DataFrame(results), x='model', y='score', text='score', color='model', color_discrete_sequence=['#10b981', '#8b5cf6', '#3b82f6'], title="Algorithm Performance (%)", template=plot_template)
                    fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=font_color, size=11), height=300)

                    imps = None
                    try:
                        if hasattr(best_model_inst, 'feature_importances_'):
                            imps = best_model_inst.feature_importances_
                        elif hasattr(best_model_inst, 'coef_'):
                            coef = np.array(best_model_inst.coef_)
                            imps = coef[0] if coef.ndim > 1 else coef
                    except Exception:
                        imps = None

                    chart_fi_html = px.bar(pd.DataFrame({'Feature': X.columns, 'Impact': np.abs(imps)}).sort_values('Impact').tail(7), x='Impact', y='Feature', orientation='h', title="Feature Importance", template=plot_template, color_discrete_sequence=['#f43f5e']).update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=font_color, size=11), height=300).to_html(full_html=False, config=plot_config) if imps is not None else "<div class='p-4 text-muted'>N/A</div>"

                    best_preds = best_model_inst.predict(X_test)
                    fig_cm = px.imshow(confusion_matrix(y_test, best_preds), text_auto=True, title="Confusion Matrix", template=plot_template, color_continuous_scale='Greens') if task_type == 'classification' else px.scatter(x=y_test, y=best_preds, labels={'x':'Actual', 'y':'Predicted'}, title="Spread", template=plot_template)
                    fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=font_color, size=11), height=300)

                    pred_features = []
                    for c in X.columns:
                        if c in encoders:
                            suggested_values = df_ml[c].mode().dropna().astype(str).tolist()[:3]
                            pred_features.append({
                                'name': c,
                                'type': 'categorical',
                                'options': encoders[c].classes_.tolist(),
                                'suggested': ', '.join(suggested_values) if suggested_values else 'Choose a category'
                            })
                        else:
                            median = df_ml[c].median()
                            mean = df_ml[c].mean()
                            pred_features.append({
                                'name': c,
                                'type': 'numeric',
                                'suggested': f"Median {median:.2f}, mean {mean:.2f}" if pd.notnull(median) else 'Enter a value'
                            })
                    context.update({
                        'prediction_features': pred_features,
                        'fit_diagnostics': {
                            'train_score': round(results[0]['train_score'], 2),
                            'test_score': round(results[0]['score'], 2),
                            'cv_score': round(results[0]['cv_score'], 2),
                            'cv_std': round(results[0].get('cv_std', 0), 2),
                            'fit_gap': round(results[0].get('fit_gap', 0), 2),
                            'status': results[0]['status']
                        },
                        'ml_results': results,
                        'best_model': results[0],
                        'metric_name': 'Accuracy' if task_type == 'classification' else 'R2 Score',
                        'target_col': target_col,
                        'task_type': task_type.capitalize(),
                        'chart_comp_html': fig_comp.to_html(full_html=False, config=plot_config),
                        'chart_fi_html': chart_fi_html,
                        'chart_cm_html': fig_cm.to_html(full_html=False, config=plot_config),
                        'prediction_features': pred_features,
                        'success_msg': "Models Trained Successfully!",
                            'fit_suggestion': fit_suggestion or 'Model selection complete.'
                    })

                # Track module usage
                if request.user.is_authenticated:
                    UserModuleUsage.objects.update_or_create(
                        user=request.user,
                        module='module3',
                        defaults={
                            'dataset_name': os.path.basename(filepath) if filepath else None,
                            'session_data': {
                                'target_column': target_col,
                                'task_type': task_type,
                                'best_model': results[0]['model'],
                                'best_score': results[0]['score'],
                                'all_results': results,
                                'feature_importance': [{'feature': X.columns[i], 'importance': float(imps[i])} for i in range(len(X.columns))] if imps is not None else None
                            }
                        }
                    )

    except Exception as e: context['error'] = f"AutoML Error: {str(e)}"
    return render(request, 'module3.html', context)


# ==========================================
# REPORT GENERATION
# ==========================================
def generate_report(request):
    if not request.user.is_authenticated:
        messages.error(request, "Please login to generate reports.")
        return redirect('login')

    # Check if user has used all modules
    used_modules = UserModuleUsage.objects.filter(user=request.user).values_list('module', flat=True)
    required_modules = {'module1', 'module2', 'module3'}

    if not required_modules.issubset(set(used_modules)):
        messages.error(request, "You must use all three modules before generating a report.")
        return redirect('home')

    # Get module usage data
    module1_data = UserModuleUsage.objects.filter(user=request.user, module='module1').first()
    module2_data = UserModuleUsage.objects.filter(user=request.user, module='module2').first()
    module3_data = UserModuleUsage.objects.filter(user=request.user, module='module3').first()

    # Generate comprehensive report data
    report_data = {
        'user': request.user.username,
        'generated_at': timezone.now().strftime('%Y-%m-%d %H:%M:%S'),
        'modules_used': list(used_modules),
        'module1': {
            'dataset_name': module1_data.dataset_name if module1_data else None,
            'operations_performed': module1_data.session_data.get('operations', {}) if module1_data else {},
            'original_stats': module1_data.session_data.get('original_stats', {}) if module1_data else {},
            'processed_stats': module1_data.session_data.get('processed_stats', {}) if module1_data else {},
        },
        'module2': {
            'dataset_name': module2_data.dataset_name if module2_data else None,
            'charts_generated': module2_data.session_data.get('total_charts', 0) if module2_data else 0,
            'pinned_charts': module2_data.session_data.get('pinned_charts', []) if module2_data else [],
        },
        'module3': {
            'dataset_name': module3_data.dataset_name if module3_data else None,
            'target_column': module3_data.session_data.get('target_column') if module3_data else None,
            'task_type': module3_data.session_data.get('task_type') if module3_data else None,
            'best_model': module3_data.session_data.get('best_model') if module3_data else None,
            'best_score': module3_data.session_data.get('best_score') if module3_data else None,
            'all_models': module3_data.session_data.get('all_results', []) if module3_data else [],
            'feature_importance': module3_data.session_data.get('feature_importance', []) if module3_data else [],
        }
    }

    # Save report to database
    user_report = UserReport.objects.create(
        user=request.user,
        report_data=report_data
    )

    # Generate PDF report
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("Insight Data Miner - Comprehensive Report", title_style))
    story.append(Spacer(1, 12))

    # User and generation info
    story.append(Paragraph(f"<b>User:</b> {report_data['user']}", styles['Normal']))
    story.append(Paragraph(f"<b>Generated:</b> {report_data['generated_at']}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Module 1: Data Preprocessing
    story.append(Paragraph("1. Data Preprocessing Module", styles['Heading2']))
    if module1_data:
        story.append(Paragraph(f"<b>Dataset:</b> {report_data['module1']['dataset_name']}", styles['Normal']))
        story.append(Paragraph("<b>Operations Performed:</b>", styles['Normal']))
        operations = report_data['module1']['operations_performed']
        for op, performed in operations.items():
            story.append(Paragraph(f"  - {op.replace('_', ' ').title()}: {'Yes' if performed else 'No'}", styles['Normal']))

        # Comparison table
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Dataset Statistics Comparison:</b>", styles['Normal']))
        comparison_data = [
            ['Metric', 'Original', 'After Processing'],
            ['Rows', report_data['module1']['original_stats'].get('rows', 'N/A'), report_data['module1']['processed_stats'].get('rows', 'N/A')],
            ['Columns', report_data['module1']['original_stats'].get('cols', 'N/A'), report_data['module1']['processed_stats'].get('cols', 'N/A')],
            ['Missing Values', report_data['module1']['original_stats'].get('missing', 'N/A'), report_data['module1']['processed_stats'].get('missing', 'N/A')],
            ['Data Health', report_data['module1']['original_stats'].get('health', 'N/A'), report_data['module1']['processed_stats'].get('health', 'N/A')],
            ['Outliers Fixed', 'N/A', report_data['module1']['processed_stats'].get('outliers_fixed', 'N/A')],
        ]

        table = Table(comparison_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No preprocessing data available.", styles['Normal']))
    story.append(Spacer(1, 20))

    # Module 2: Visual Analytics
    story.append(Paragraph("2. Visual Analytics Module", styles['Heading2']))
    if module2_data:
        story.append(Paragraph(f"<b>Dataset:</b> {report_data['module2']['dataset_name']}", styles['Normal']))
        story.append(Paragraph(f"<b>Total Charts Generated:</b> {report_data['module2']['charts_generated']}", styles['Normal']))
        story.append(Paragraph(f"<b>Pinned Charts:</b> {len(report_data['module2']['pinned_charts'])}", styles['Normal']))

        if report_data['module2']['pinned_charts']:
            story.append(Paragraph("<b>Chart Details:</b>", styles['Normal']))
            for i, chart in enumerate(report_data['module2']['pinned_charts'], 1):
                story.append(Paragraph(f"  {i}. Type: {chart['type']}, X: {chart.get('x', 'N/A')}, Y: {chart.get('y', 'N/A')}", styles['Normal']))
    else:
        story.append(Paragraph("No visual analytics data available.", styles['Normal']))
    story.append(Spacer(1, 20))

    # Module 3: AutoML
    story.append(Paragraph("3. AutoML Module", styles['Heading2']))
    if module3_data:
        story.append(Paragraph(f"<b>Dataset:</b> {report_data['module3']['dataset_name']}", styles['Normal']))
        story.append(Paragraph(f"<b>Target Column:</b> {report_data['module3']['target_column']}", styles['Normal']))
        story.append(Paragraph(f"<b>Task Type:</b> {report_data['module3']['task_type']}", styles['Normal']))
        story.append(Paragraph(f"<b>Best Model:</b> {report_data['module3']['best_model']} ({report_data['module3']['best_score']})", styles['Normal']))

        # Model performance table
        if report_data['module3']['all_models']:
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Model Performance Comparison:</b>", styles['Normal']))
            model_data = [['Model', 'Score']]
            for model in report_data['module3']['all_models']:
                model_data.append([model['model'], f"{model['score']}"])

            model_table = Table(model_data)
            model_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(model_table)

        # Feature importance
        if report_data['module3']['feature_importance']:
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Top Feature Importance:</b>", styles['Normal']))
            for feature in sorted(report_data['module3']['feature_importance'], key=lambda x: x['importance'], reverse=True)[:5]:
                story.append(Paragraph(f"  - {feature['feature']}: {feature['importance']:.4f}", styles['Normal']))
    else:
        story.append(Paragraph("No AutoML data available.", styles['Normal']))

    doc.build(story)
    buffer.seek(0)

    # Return PDF response
    response = FileResponse(buffer, as_attachment=True, filename=f'IDM_Report_{request.user.username}_{timezone.now().strftime("%Y%m%d_%H%M%S")}.pdf')
    user_report.is_downloaded = True
    user_report.save()
    return response


def check_report_eligibility(request):
    """AJAX endpoint to check if user can generate report"""
    if not request.user.is_authenticated:
        return JsonResponse({'eligible': False, 'message': 'Please login first.'})

    used_modules = UserModuleUsage.objects.filter(user=request.user).values_list('module', flat=True)
    required_modules = {'module1', 'module2', 'module3'}
    eligible = required_modules.issubset(set(used_modules))

    return JsonResponse({
        'eligible': eligible,
        'message': 'Report ready for download!' if eligible else f'Complete {len(required_modules - set(used_modules))} more modules.',
        'used_modules': list(used_modules)
    })