import os
import io
import base64
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

from django import forms
from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate

# ==========================================
# STRICT CUSTOM SIGNUP FORM
# ==========================================
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
            if password != confirm_password:
                self.add_error('confirm_password', "Passwords do not match!")
            if len(password) < 8:
                self.add_error('password', "Must be at least 8 characters.")
            if not re.search(r'[A-Z]', password):
                self.add_error('password', "Must contain an Uppercase letter.")
            if not re.search(r'[a-z]', password):
                self.add_error('password', "Must contain a Lowercase letter.")
            if not re.search(r'\d', password):
                self.add_error('password', "Must contain a Number.")
            if not re.search(r'[@$!%*?&#]', password):
                self.add_error('password', "Must contain a Special Character (@$!%*?&#).")
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password"])
        if commit:
            user.save()
        return user


# ==========================================
# 1. LANDING PAGE & AUTHENTICATION VIEWS
# ==========================================
def home(request):
    return render(request, 'upload.html')

def signup_view(request):
    if request.method == 'POST':
        form = ModernSignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = ModernSignupForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = AuthenticationForm()
    for field in form.fields.values():
        field.widget.attrs['class'] = 'form-control'
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('home')


# ==========================================
# 2. MODULE 1 (PREPROCESSING)
# ==========================================
def module1_workspace(request):
    context = {}
    fs = FileSystemStorage()

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'upload_file' and request.FILES.get('dataset'):
            uploaded_file = request.FILES['dataset']
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            allowed = ['.csv', '.xlsx', '.xls']
            
            if ext not in allowed:
                context['error'] = "Unsupported Format! Please provide a valid .csv or .excel file."
            else:
                filename = fs.save(uploaded_file.name, uploaded_file)
                file_path = fs.path(filename)
                request.session['active_dataset'] = file_path
                context['filename'] = filename
                context['message'] = "Dataset successfully ingested into the Intelligence Engine."

        elif action == 'execute_pipeline':
            filename = request.POST.get('filename')
            file_path = os.path.join(settings.MEDIA_ROOT, filename)
            try:
                if filename.endswith('.csv'): df = pd.read_csv(file_path)
                else: df = pd.read_excel(file_path)

                orig_rows, orig_cols = df.shape
                orig_missing = int(df.isnull().sum().sum())
                orig_memory = df.memory_usage(deep=True).sum() / 1024 
                orig_cats = len(df.select_dtypes(include=['object', 'category']).columns)
                orig_health = round(100 * (1 - (orig_missing / (orig_rows * orig_cols))), 1) if (orig_rows * orig_cols) > 0 else 0
                outliers_fixed = 0

                remove_input = request.POST.get('remove_cols', '').strip()
                if remove_input:
                    cols_to_del = [c.strip() for c in remove_input.split(',') if c.strip() in df.columns]
                    df.drop(columns=cols_to_del, inplace=True)

                rename_input = request.POST.get('rename_cols', '').strip()
                if rename_input:
                    try:
                        rename_map = dict(item.split(':') for item in rename_input.split(',') if ':' in item)
                        df.rename(columns={k.strip(): v.strip() for k, v in rename_map.items()}, inplace=True)
                    except: pass

                if request.POST.get('do_clean'):
                    threshold = 0.5
                    df.drop(columns=[c for c in df.columns if df[c].isnull().sum()/len(df) > threshold], inplace=True)
                    for col in df.columns:
                        if df[col].isnull().sum() > 0:
                            if df[col].dtype in ['int64', 'float64']:
                                df[col] = df[col].fillna(df[col].median())
                            else:
                                if not df[col].mode().empty:
                                    df[col] = df[col].fillna(df[col].mode()[0])

                plot_data = []
                if request.POST.get('do_outliers'):
                    nums = df.select_dtypes(include=[np.number]).columns[:4]
                    for col in nums:
                        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                        
                        outliers_fixed += len(df[(df[col] < lower) | (df[col] > upper)])
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                        sns.boxplot(x=df[col], ax=ax1, color='#ef4444') 
                        ax1.set_title('Outliers Detected')
                        
                        df[col] = np.clip(df[col], lower, upper)
                        
                        sns.boxplot(x=df[col], ax=ax2, color='#10b981')
                        ax2.set_title('Outliers Removed')
                        
                        buf = io.BytesIO()
                        plt.tight_layout()
                        plt.savefig(buf, format='png', transparent=True)
                        buf.seek(0)
                        plot_data.append({'column': col, 'chart': base64.b64encode(buf.read()).decode('utf-8')})
                        plt.close(fig)

                if request.POST.get('do_encode'):
                    for col in df.select_dtypes(include=['object', 'category']).columns:
                        df[col] = df[col].astype('category').cat.codes

                if request.POST.get('do_scale'):
                    nums = df.select_dtypes(include=[np.number]).columns
                    for col in nums:
                        if df[col].std() != 0:
                            df[col] = (df[col] - df[col].mean()) / df[col].std()

                final_rows, final_cols = df.shape
                final_missing = int(df.isnull().sum().sum())
                final_memory = df.memory_usage(deep=True).sum() / 1024
                final_cats = len(df.select_dtypes(include=['object', 'category']).columns)
                final_health = 100.0 if final_missing == 0 else round(100 * (1 - (final_missing / (final_rows * final_cols))), 1)

                df.to_csv(file_path, index=False)
                request.session['active_dataset'] = file_path
                
                context.update({
                    'show_results': True, 'plot_data': plot_data, 'filename': filename,
                    'comparison': {
                        'before': {
                            'rows': orig_rows, 'cols': orig_cols, 'missing': orig_missing,
                            'memory': f"{orig_memory:.1f} KB", 'cats': orig_cats, 'health': f"{orig_health}%"
                        },
                        'after': {
                            'rows': final_rows, 'cols': final_cols, 'missing': final_missing,
                            'memory': f"{final_memory:.1f} KB", 'cats': final_cats, 'health': f"{final_health}%",
                            'outliers_fixed': outliers_fixed
                        }
                    }
                })
            except Exception as e:
                context['error'] = f"Processing Error: {str(e)}"

    if 'filename' in context:
        try:
            file_path = os.path.join(settings.MEDIA_ROOT, context['filename'])
            df_pv = pd.read_csv(file_path)
            context.update({
                'total_rows': df_pv.shape[0], 'total_cols': df_pv.shape[1],
                'total_missing': int(df_pv.isnull().sum().sum()),
                'data_html': df_pv.head(5).to_html(classes='table table-dark table-hover mb-0', index=False, border=0)
            })
        except: pass

    return render(request, 'module1.html', context)


# ==========================================
# 3. MODULE 2 (VISUAL ANALYTICS & DASHBOARD)
# ==========================================
def module2_visual_analytics(request):
    context = {'module': 'module2'}
    
    if 'pinned_charts' not in request.session:
        request.session['pinned_charts'] = []
    
    context['auto_summary'] = request.session.get('auto_summary', None)
    context['auto_dashboard_charts'] = request.session.get('auto_dashboard_charts', None)
    
    if request.method == 'POST' and 'new_dataset' in request.FILES:
        file = request.FILES['new_dataset']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        filepath = fs.path(filename)
        request.session['active_dataset'] = filepath 
        request.session['pinned_charts'] = []
        if 'auto_summary' in request.session: del request.session['auto_summary']
        if 'auto_dashboard_charts' in request.session: del request.session['auto_dashboard_charts']
        request.session.modified = True
        context['success_msg'] = "New Dataset Loaded Successfully!"
        context['auto_summary'] = None
        context['auto_dashboard_charts'] = None
    
    filepath = request.session.get('active_dataset', None)
    
    if not filepath or not os.path.exists(filepath):
        context['standby'] = True
        return render(request, 'module2.html', context)
        
    try:
        if filepath.endswith('.csv'): df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')): df = pd.read_excel(filepath)
        else:
            context['error'] = "Unsupported file format."
            return render(request, 'module2.html', context)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        all_cols = df.columns.tolist()
        
        context.update({
            'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols,
            'all_cols': all_cols, 'dataset_name': os.path.basename(filepath),
            'total_rows': len(df)
        })

        # Config for Manual Builder (Keep Toolbars)
        manual_plot_config = {'displaylogo': False, 'displayModeBar': True}
        # Config for Auto Dashboard (NO TOOLBARS - Clean Look)
        auto_plot_config = {'displaylogo': False, 'displayModeBar': False}

        if request.user.is_authenticated:
            manual_plot_config['toImageButtonOptions'] = {
                'format': 'png', 'filename': 'InsightDataMiner_HD_Visual',
                'height': 600, 'width': 800, 'scale': 2
            }

        def get_plotly_fig(c_type, x, y, color, custom_height=None, is_auto=False):
            fig = None
            try:
                x = x if x else None
                y = y if y else None
                
                # Use a tighter margin for auto dashboard
                margin_dict = dict(l=10, r=10, t=30, b=10) if is_auto else dict(l=20, r=20, t=40, b=20)
                
                if c_type == 'scatter' and x and y: fig = px.scatter(df, x=x, y=y, color=color, template="plotly_dark", title=f"{y} vs {x}")
                elif c_type == 'bar' and x and y: fig = px.bar(df, x=x, y=y, color=color, template="plotly_dark", barmode='group', title=f"{y} by {x}")
                elif c_type == 'line' and x and y: fig = px.line(df, x=x, y=y, color=color, template="plotly_dark", title=f"Trend: {y} over {x}")
                elif c_type == 'histogram' and x: fig = px.histogram(df, x=x, color=color, template="plotly_dark", title=f"Distribution of {x}")
                elif c_type == 'box' and y: fig = px.box(df, x=x, y=y, color=color, template="plotly_dark", title=f"Box Plot of {y}")
                elif c_type == 'pie' and x: 
                    if y: fig = px.pie(df, names=x, values=y, hole=0.5, template="plotly_dark", title=f"{y} by {x}")
                    else: fig = px.pie(df, names=x, hole=0.5, template="plotly_dark", title=f"Distribution of {x}")
                elif c_type == 'heatmap': fig = px.imshow(df[numeric_cols].corr(), text_auto=True, aspect="auto", template="plotly_dark", color_continuous_scale='Viridis', title="Correlation Heatmap")
                
                if fig:
                    h = custom_height if custom_height else 500
                    fig.update_layout(height=h, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8', size=11), margin=margin_dict)
            except Exception as e: 
                pass
            return fig

        if request.method == 'POST':
            action = request.POST.get('action')
            
            if action == 'clear_dashboard':
                request.session['pinned_charts'] = []
                if 'auto_summary' in request.session: del request.session['auto_summary']
                if 'auto_dashboard_charts' in request.session: del request.session['auto_dashboard_charts']
                context['auto_summary'] = None
                context['auto_dashboard_charts'] = None
                request.session.modified = True
                context['success_msg'] = "Dashboard cleared successfully!"
                
            # ========================================================
            # 🌟 SMART UNIVERSAL AUTO-DASHBOARD LOGIC (BUSINESS CLASS)
            # ========================================================
            elif action == 'auto_dashboard':
                # 1. Smart Down-sampling (Speed Boost)
                df_plot = df.sample(n=2000, random_state=42) if len(df) > 2000 else df
                
                # 2. Heuristic Selection (Ignore IDs, sort by fewest categories)
                valid_cats = [c for c in categorical_cols if 1 < df[c].nunique() <= 10 and 'id' not in c.lower() and 'name' not in c.lower()]
                valid_nums = [c for c in numeric_cols if 'id' not in c.lower() and 'index' not in c.lower()]
                
                valid_cats = sorted(valid_cats, key=lambda x: df[x].nunique()) # Target variable is usually binary/fewest options
                
                cat_target = valid_cats[0] if valid_cats else None
                cat_group = valid_cats[1] if len(valid_cats) > 1 else cat_target
                
                num_main = valid_nums[0] if valid_nums else None
                num_sec = valid_nums[1] if len(valid_nums) > 1 else num_main

                # 3. Generating TRUE Analytical KPIs
                summary = {}
                
                # KPI 1: Main Target Distribution
                if cat_target:
                    vc = df[cat_target].value_counts(normalize=True) * 100
                    top_class = vc.index[0]
                    top_pct = round(vc.iloc[0], 1)
                    summary['kpi1_title'] = f"Primary Target: {cat_target}"
                    summary['kpi1_val'] = f"{top_class} ({top_pct}%)"
                    summary['kpi1_sub'] = "Dominant Category"
                else:
                    summary['kpi1_title'] = "Volume Metric"
                    summary['kpi1_val'] = f"{len(df)} Rows"
                    summary['kpi1_sub'] = "Total Observations"

                # KPI 2: Cross Impact (Real Insight)
                if cat_target and num_main and cat_target != cat_group:
                    try:
                        # Find which group has the highest average of the main numeric
                        best_group = df.groupby(cat_group)[num_main].mean().idxmax()
                        max_mean = round(df.groupby(cat_group)[num_main].mean().max(), 1)
                        summary['kpi2_title'] = f"Impact on {num_main}"
                        summary['kpi2_val'] = f"Peak: {max_mean}"
                        summary['kpi2_sub'] = f"Highest in group '{best_group}'"
                    except:
                        summary['kpi2_title'] = f"Metric: {num_main}"
                        summary['kpi2_val'] = f"Avg: {round(df[num_main].mean(), 1)}"
                        summary['kpi2_sub'] = "Overall Average"
                else:
                    summary['kpi2_title'] = "Features Count"
                    summary['kpi2_val'] = f"{len(df.columns)} Cols"
                    summary['kpi2_sub'] = "Available Attributes"

                # KPI 3: Numeric Baseline
                if num_main:
                    avg_n = round(df[num_main].mean(), 2)
                    summary['kpi3_title'] = f"Baseline: {num_main}"
                    summary['kpi3_val'] = f"Avg: {avg_n}"
                    summary['kpi3_sub'] = "Overall Dataset Mean"
                else:
                    summary['kpi3_title'] = "Baseline Metric"
                    summary['kpi3_val'] = "N/A"
                    summary['kpi3_sub'] = "No valid numerics"

                # KPI 4: Numeric Spread / Max
                if num_sec:
                    max_n = round(df[num_sec].max(), 2)
                    summary['kpi4_title'] = f"Peak: {num_sec}"
                    summary['kpi4_val'] = f"Max: {max_n}"
                    summary['kpi4_sub'] = "Highest recorded value"
                else:
                    summary['kpi4_title'] = "Peak Value"
                    summary['kpi4_val'] = "N/A"
                    summary['kpi4_sub'] = "N/A"

                # 4. Generate Compact Bento-Box Graphs (Height: 220px to fit on 1 screen)
                charts_html = []
                # Use auto_plot_config here to HIDE TOOLBARS
                bento_layout = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8', size=11), margin=dict(l=10, r=10, t=30, b=10), height=220)
                
                try:
                    if cat_target:
                        f1 = px.pie(df_plot, names=cat_target, hole=0.5, template="plotly_dark", title=f"Distribution of {cat_target}")
                        f1.update_layout(**bento_layout)
                        charts_html.append(f1.to_html(full_html=False, include_plotlyjs='cdn', config=auto_plot_config))
                    if cat_group and num_main:
                        f2 = px.histogram(df_plot, x=cat_group, y=num_main, color=cat_target if cat_target!=cat_group else None, barmode='group', histfunc='avg', template="plotly_dark", title=f"Avg {num_main} by {cat_group}")
                        f2.update_layout(**bento_layout)
                        charts_html.append(f2.to_html(full_html=False, include_plotlyjs='cdn', config=auto_plot_config))
                    if num_main:
                        f3 = px.histogram(df_plot, x=num_main, color=cat_target if cat_target else None, template="plotly_dark", title=f"{num_main} Density Spread")
                        f3.update_layout(**bento_layout)
                        charts_html.append(f3.to_html(full_html=False, include_plotlyjs='cdn', config=auto_plot_config))
                    if num_main and num_sec:
                        f4 = px.scatter(df_plot, x=num_main, y=num_sec, color=cat_target if cat_target else None, template="plotly_dark", title=f"{num_main} vs {num_sec}")
                        f4.update_layout(**bento_layout)
                        charts_html.append(f4.to_html(full_html=False, include_plotlyjs='cdn', config=auto_plot_config))
                except Exception as e:
                    pass

                request.session['auto_summary'] = summary
                request.session['auto_dashboard_charts'] = charts_html
                context['auto_summary'] = summary
                context['auto_dashboard_charts'] = charts_html
                context['success_msg'] = "Business Class Dashboard Generated Successfully!"

            elif action in ['generate_chart', 'pin_chart']:
                c_type = request.POST.get('chart_type')
                x_col = request.POST.get('x_axis')
                y_col = request.POST.get('y_axis')
                c_col = request.POST.get('color_col')
                if c_col == 'none': c_col = None
                
                chart_settings = {'type': c_type, 'x': x_col, 'y': y_col, 'color': c_col}

                if action == 'pin_chart':
                    pinned = request.session.get('pinned_charts', [])
                    pinned.append(chart_settings)
                    request.session['pinned_charts'] = pinned
                    request.session.modified = True
                    context['success_msg'] = "Chart Pinned to Dashboard successfully!"

                # Manual charts keep the toolbar (manual_plot_config)
                main_fig = get_plotly_fig(c_type, x_col, y_col, c_col, custom_height=450, is_auto=False)
                if main_fig:
                    context['chart_html'] = main_fig.to_html(full_html=False, include_plotlyjs='cdn', config=manual_plot_config)
                    context['current_config'] = chart_settings
                else:
                    context['chart_error'] = "Invalid parameters selected for this chart type."

        # Render Pinned Dashboard Grid (Keeps toolbar)
        dashboard_html = []
        for p in request.session.get('pinned_charts', []):
            p_fig = get_plotly_fig(p['type'], p['x'], p['y'], p['color'], custom_height=300, is_auto=False)
            if p_fig:
                dashboard_html.append(p_fig.to_html(full_html=False, include_plotlyjs='cdn', config=manual_plot_config))
        
        context['dashboard_charts'] = dashboard_html

    except Exception as e:
        context['error'] = f"Engine Error: {str(e)}"
        
    return render(request, 'module2.html', context)


# ==========================================
# 4. MODULE 3 (AUTO-ML PREDICTIONS)
# ==========================================
def module3_automl(request):
    context = {'module': 'module3'}
    
    if request.method == 'POST' and 'new_dataset' in request.FILES:
        file = request.FILES['new_dataset']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        filepath = fs.path(filename)
        request.session['active_dataset'] = filepath 
        context['success_msg'] = "Dataset Loaded for ML Engine!"
    
    filepath = request.session.get('active_dataset', None)
    
    if not filepath or not os.path.exists(filepath):
        context['standby'] = True
        return render(request, 'module3.html', context)
        
    try:
        if filepath.endswith('.csv'): df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')): df = pd.read_excel(filepath)
        else:
            context['error'] = "Unsupported file format. Please upload a .csv or .excel file."
            return render(request, 'module3.html', context)
        
        context['all_cols'] = df.columns.tolist()
        context['dataset_name'] = os.path.basename(filepath)
        context['total_rows'] = len(df)

        if request.method == 'POST' and request.POST.get('action') == 'run_automl':
            target_col = request.POST.get('target_col')
            task_type = request.POST.get('task_type')
            
            if not target_col or not task_type:
                context['error'] = "Please select both Target Feature and Task Type."
            else:
                df_ml = df.dropna().copy()
                
                if len(df_ml) < 50:
                    context['error'] = "Not enough clean data to train models. Please use Module 1 to impute missing values."
                    return render(request, 'module3.html', context)

                X = df_ml.drop(columns=[target_col])
                y = df_ml[target_col]

                for col in X.select_dtypes(include=['object', 'category']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

                if task_type == 'classification' and (y.dtype == 'object' or y.dtype.name == 'category'):
                    le_y = LabelEncoder()
                    y = le_y.fit_transform(y.astype(str))

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                results = []
                if task_type == 'classification':
                    models = {
                        'Logistic Regression': LogisticRegression(max_iter=1000),
                        'Decision Tree': DecisionTreeClassifier(random_state=42),
                        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
                    }
                    metric_name = 'Accuracy'
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        score = accuracy_score(y_test, preds) * 100
                        results.append({'model': name, 'score': round(score, 2)})
                else: 
                    models = {
                        'Linear Regression': LinearRegression(),
                        'Decision Tree': DecisionTreeRegressor(random_state=42),
                        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
                    }
                    metric_name = 'R2 Score'
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        score = r2_score(y_test, preds) * 100
                        score = max(0, score) 
                        results.append({'model': name, 'score': round(score, 2)})

                results = sorted(results, key=lambda x: x['score'], reverse=True)
                best_model = results[0]

                df_res = pd.DataFrame(results)
                fig = px.bar(
                    df_res, 
                    x='model', 
                    y='score', 
                    text='score', 
                    color='model',
                    color_discrete_sequence=['#10b981', '#a855f7', '#ec4899'], 
                    title=f"AI Models Comparison ({metric_name} %)", 
                    template="plotly_dark"
                )
                
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                
                plot_config = {'displaylogo': False}
                if not request.user.is_authenticated:
                    plot_config['modeBarButtonsToRemove'] = ['toImage']
                    context['guest_warning'] = True

                context.update({
                    'ml_results': results,
                    'best_model': best_model,
                    'metric_name': metric_name,
                    'target_col': target_col,
                    'task_type': task_type.capitalize(),
                    'chart_html': fig.to_html(full_html=False, include_plotlyjs='cdn', config=plot_config)
                })

    except Exception as e:
        context['error'] = f"AutoML Engine Error: {str(e)}"
        
    return render(request, 'module3.html', context)