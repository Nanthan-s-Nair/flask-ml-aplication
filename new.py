import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv('/Users/nanthansnair/Downloads/jupiter/data1.csv')
df['Actual flow volume air/gas'] = df['Actual flow volume air/gas'].str.replace(',', '').str.replace(' m3/h', '').astype(float)
df['Pressure, static'] = df['Pressure, static'].str.replace(',', '').str.replace(' Pa', '').astype(float)
df['Rated power'] = df['Rated power'].str.replace(' kW', '').astype(float)
df.drop(columns=['Unnamed: 5'], inplace=True)
df.dropna(inplace=True)

X = df[['Actual flow volume air/gas', 'Pressure, static']]
y = df['Rated power']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

def recommend_fan(flow, pressure):
    input_scaled = scaler.transform(np.array([[flow, pressure]]))
    predicted_power = model.predict(input_scaled)[0][0]
    df['Power Difference'] = abs(df['Rated power'] - predicted_power)
    sorted_fans = df.sort_values(by='Power Difference', ascending=True)
    best_fan = sorted_fans.iloc[0]
    return best_fan.to_dict()


app = dash.Dash(__name__)

app.layout = html.Div([
    html.Header([
        html.Img(src='/static/logo.png', className='logo',style={'height': '50px','width' : '150px','margin-left' : '-1310px'}),
        html.H1('Enter The Values of Parameters', style={'color': '#fff'}),
    ], style={
        'background-color' : '#3f51b5','padding' : '10px','color' : '#fff', 
        'text-align' : 'center','height': '55px','position' : 'fixed','width' : '100%',
        'top' : '0',
        'z-index': '1000',
    }),
    
    html.Main([
        html.Div([
            html.Div([
                html.Label('Flow (m3/h): '),
                dcc.Input(id='flow', type='number', required=True)
            ]),
            html.Div([
                html.Label('Pressure (Pa): '),
                dcc.Input(id='pressure', type='number', required=True)
            ]),
            html.Button('Recommend Fan', id='submit-button', n_clicks=0)
        ], style={'margin': '20px', 'textAlign': 'center'}),
        
        html.Div(id='recommended-fan', style={'textAlign': 'center', 'marginTop': '20px'}),
        
        html.H2('Optional Parameters', style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Label('Type'),
                dcc.Input(id='optional1', type='text')
            ]),
            html.Div([
                html.Label('Op. temp. (°C)'),
                dcc.Input(id='optional2', type='number')
            ]),
            html.Div([
                html.Label('VFD'),
                dcc.Input(id='optional3', type='text')
            ]),
            html.Div([
                html.Label('Type of drive'),
                dcc.Input(id='optional4', type='text')
            ]),
            html.Div([
                html.Label('Outlet orientation / Design'),
                dcc.Input(id='optional7', type='text')
            ]),
            html.Div([
                html.Label('Material specification'),
                dcc.Input(id='optional8', type='text')
            ]),
            html.Div([
                html.Label('Material impeller'),
                dcc.Input(id='optional9', type='text')
            ]),
            html.Div([
                html.Label('Efficiency class'),
                dcc.Input(id='optional10', type='text')
            ]),
        ], className='form1', style={
            'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'rowGap': '19px', 'columnGap': '15px', 'justifyItems': 'center'
        }),
        
        html.Div([
            html.Div([dcc.Graph(id='graph1')], className='graph-container'),
            html.Div([dcc.Graph(id='graph2')], className='graph-container'),
        ], className='graph-gallery', style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}),
        
    ], style={'padding': '20px', 'marginTop': '60px'}),
    
    html.Footer([
        html.Img(src='/static/footer1.png', className='logo')
    ], style={
        'backgroundColor': '#3f51b5', 'color': '#fff', 'padding': '10px', 
        'textAlign': 'center', 'position': 'static', 'width': '100%', 'bottom': '0'
    })
])

@app.callback(
    Output('recommended-fan', 'children'),
    Input('submit-button', 'n_clicks'),
    State('flow', 'value'),
    State('pressure', 'value')
)
def update_recommendation(n_clicks, flow, pressure):
    if n_clicks == 0:
        raise PreventUpdate
    try:
        recommended_fan = recommend_fan(flow, pressure)
        return html.Div([
            html.H3('Recommended Fan:'),
            html.Pre(recommended_fan)
        ])
    except Exception as e:
        return html.Div(f"Error: {str(e)}")

@app.callback(
    Output('graph1', 'figure'),
    Output('graph2', 'figure'),
    Input('submit-button', 'n_clicks'),
    State('flow', 'value'),
    State('pressure', 'value')
)
def update_graphs(n_clicks, flow, pressure):
    if n_clicks == 0:
        raise PreventUpdate
    try:
        fig1 = px.scatter(df, x='Actual flow volume air/gas', y='Rated power', color='Pressure, static', title='Flow vs. Power')
        fig2 = px.scatter(df, x='Pressure, static', y='Rated power', color='Actual flow volume air/gas', title='Pressure vs. Power')
        return fig1, fig2
    except Exception as e:
        fig = px.scatter(title=f"Error: {str(e)}")
        return fig, fig

if __name__ == '__main__':
    app.run_server(debug=True, port=5001)
