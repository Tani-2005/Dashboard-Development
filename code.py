import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from dash import Dash, dcc, html

data = pd.read_csv('ntrarogyaseva.csv')
data = data.dropna()
data = pd.get_dummies(data, drop_first=True)

target_column = 'AGE'
if target_column not in data.columns:
    raise ValueError(f"Target column '{target_column}' not found in the dataset.")

X = data.drop(target_column, axis=1)
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

violin_fig = px.violin(data, y='AGE', box=True, points='all', title="Violin Plot of AGE")

feature_line = 'PREAUTH_AMT' 
if feature_line in data.columns:
    line_fig = px.line(data, x=data.index, y=feature_line,
                       title=f"Line Graph of {feature_line}")
else:
    line_fig = None
  
pie_chart_fig = px.pie(names=y.value_counts().index,
                       values=y.value_counts().values,
                       title="Target Variable Distribution")

scatter_feature_1, scatter_feature_2 = 'PREAUTH_AMT', 'CLAIM_AMOUNT'
if scatter_feature_1 in X.columns and scatter_feature_2 in X.columns:
    scatter_plot_fig = px.scatter(x=X[scatter_feature_1],
                                   y=X[scatter_feature_2],
                                   color=y,
                                   title=f"Scatter Plot of {scatter_feature_1} vs {scatter_feature_2}",
                                   labels={scatter_feature_1: scatter_feature_1, scatter_feature_2: scatter_feature_2})
else:
    scatter_plot_fig = None

feature_bar = 'AGE'
if feature_bar in data.columns:
    bar_fig = px.bar(data[feature_bar].value_counts(),
                     x=data[feature_bar].value_counts().index,
                     y=data[feature_bar].value_counts().values,
                     labels={'x': feature_bar, 'y': 'Count'},
                     title=f"Bar Graph of {feature_bar}")
else:
    bar_fig = None

feature_hist = 'AGE'
if feature_hist in data.columns:
    hist_fig = px.histogram(data, x=feature_hist, nbins=20, title=f"Histogram of {feature_hist}")
else:
    hist_fig = None

app = Dash(__name__)
app.layout = html.Div([
    html.H1("Interactive Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.H2("Violin Plot of AGE"),
        dcc.Graph(figure=violin_fig)
    ]),
    html.Div([
        html.H2(f"Line Graph of {feature_line}"),
        dcc.Graph(figure=line_fig) if line_fig else html.Div("Line graph feature not found.")
    ]),

    html.Div([
        html.H2("Target Variable Distribution"),
        dcc.Graph(figure=pie_chart_fig)
    ]),
    html.Div([
        html.H2(f"Scatter Plot of {scatter_feature_1} vs {scatter_feature_2}"),
        dcc.Graph(figure=scatter_plot_fig) if scatter_plot_fig else html.Div("Scatter plot features not found.")
    ]),
    html.Div([
        html.H2(f"Bar Graph of {feature_bar}"),
        dcc.Graph(figure=bar_fig) if bar_fig else html.Div("Bar graph feature not found.")
    ]),
    html.Div([
        html.H2(f"Histogram of {feature_hist}"),
        dcc.Graph(figure=hist_fig) if hist_fig else html.Div("Histogram feature not found.")
    ])
])
if __name__ == '__main__':
    app.run_server(debug=True)
