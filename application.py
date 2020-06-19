#!/usr/bin/env python
# coding: utf-8

# # Initial Setup

# In[1]:


import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.decomposition import PCA

from sklearn import metrics

import cufflinks as cf


# # Dataset including historical data for Wheat, Barley and Maize

# In[2]:


dataset = pd.read_csv('data/Wheat_Barley_Maize_06112020.csv', delimiter = ',')
#dataset.describe()



pre_defaults = dataset.fillna(method='ffill')


# In[5]:


dataset = dataset.fillna(method='ffill')
dataset = dataset.fillna(method='bfill')
dataset.isnull().any()




X_names = [  'US (Gulf), Wheat (US No. 2, Soft Red Winter)',
    'Argentina, Wheat (Argentina, Trigo Pan, Up River, f.o.b.)',
    'Australia (Eastern States), Wheat (ASW)',
    'Australia (Eastern States), Barley (feed)',
    'Argentina, Maize (Argentina, Up River, f.o.b.)',
    'Black Sea, Barley (feed)', 
    'Black Sea, Maize (feed)',
 #   'Black Sea, Wheat (feed)',
 #   'Black Sea, Wheat (milling)',
 #   'Canada (St Lawrence), Wheat (CWRS)',
    'Brazil (Paranagua), Maize (feed)',
   # 'EU (France), Barley (feed)', #
    #'EU (France), Wheat (grade 1)', #
    'EU (UK), Wheat (feed)',
    'Kazakhstan, Wheat (milling, d.a.p. Saryagash station)',
    'Kazakhstan, Wheat (milling, f.o.b. Aktau port)',
    'Russian Federation, Wheat (milling, offer, f.o.b., deep-sea ports)',
  #  'US (Gulf), Maize (US No. 2, Yellow)', #
   # 'US (Gulf), Wheat (US No. 2, Hard Red Winter)', #
    'US (Kentucky, KY), Maize (US No. 2, White)',
    'Ukraine, Maize (offer, f.o.b.)',
    'Ukraine, Wheat (milling, offer, f.o.b.)'
    ]


X = dataset[X_names].values


# In[9]:


Y = dataset['Black Sea, Wheat (milling)'].values


# # Testing Principle Component Analysis Opportunities

# In[10]:


pca = PCA(n_components=11)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)


# # Determining Average Historical Price For Later Analysis

# In[11]:


BSWheatMillMean = dataset['Black Sea, Wheat (milling)'].mean()
BSWheatMillMean


# In[12]:


# # Splitting Data Into Training and Testing Sets

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
Xpca_train, Xpca_test, ypca_train, ypca_test = train_test_split(X_pca, Y, test_size=0.2, random_state=0)


# # Regressor selection
# ## After testing a number of potential algorithms, Bayesian Ridge has provided very good predictive value as will be seen below

# In[14]:


#regressor = LinearRegression()  
regressor = BayesianRidge()
regressor.fit(X_train, y_train)


# In[15]:


y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(10)



# In[16]:


regressorPca = BayesianRidge()
regressorPca.fit(Xpca_train, ypca_train)


# In[17]:


ypca_pred = regressorPca.predict(Xpca_test)
dfPca = pd.DataFrame({'Actual': ypca_test, 'Predicted': ypca_pred})
df1Pca = dfPca.head(10)





rgCF = regressor.coef_
#

rgCF


# In[21]:



rgCFM = pd.DataFrame(rgCF, X_names ).abs()

#rgCFM = pd.crosstab(rgCFM, rgCFM)
rgCFM



import dash
import dash_table
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import plotly.express as px
from plotly.offline import iplot

USER_AUTH = [
    ['user', 'pass123']
]

#stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
stylesheet = [#'https://github.com/plotly/dash-app-stylesheets/blob/master/dash-analytics-report.css',
                'https://codepen.io/chriddyp/pen/bWLwgP.css',
             'https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap-grid.min.css']

app = dash.Dash(external_stylesheets=stylesheet)
application = app.server

colors = {
    'background': '#ececec',
    'text': '#7FDBFF'
}
    
auth = dash_auth.BasicAuth(
    app, 
    USER_AUTH
)

app.title = 'Black Sea Wheat Values'

ds_melt= pd.melt(pre_defaults, id_vars=['Date-Monthly'], var_name='Commodity', value_name='USD/Ton')

fig = px.line(ds_melt,  x="Date-Monthly", y="USD/Ton", title='Historical Commodity Prices', color="Commodity")
fig.show()

corrFig = px.imshow(rgCFM, title="Regional Price Correlations", labels=dict(x="Commodity", color="Correlation"), y=X_names, width=900, height=600)
corrFig.show()

# Also show coefficients as well as an exploration section

def input_builder(names):
    count = 1
    
    modified_component = html.Div()
    
    for name in names:
        idTag = "comm_%d" % (count)
        print("Checking ", name)
        new_component =  dcc.Input(id=idTag,
            placeholder='Enter commodity price',
            type='number',
            value='' )
        count = count+1
        modified_component = modified_component.append(new_component)
    return modified_component

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    
   html.Div([
        dbc.Row([]),
        dbc.Row([
            dbc.Col(width=1),
            dbc.Col([
                html.H1('Prediction Tool for Black Sea Wheat Prices')], width=8)
        ]),
       
        dbc.Row([
            dbc.Col(width=2),
            dbc.Col([
               html.Div('Draw a box on the graph to zoom in, double click to reset.')], width=8)
         ]),  
        dbc.Row([]),

    
   ]), 
    
    dcc.Graph(figure=fig), 
    
    dcc.Graph(figure=corrFig),
        
    html.Div([
    
      dcc.Markdown('''
#### Prediction Section for Black Sea Milling Wheat Prices

Each neighboring region's prices have been defaulted to the last value available for that region (USD/Ton)

To predict the market value for Black Sea Milling Wheat as a function of the following significant features, modify the price and click Submit
''') ]),
    
    html.Div([
    
    # Would of course prefer to do this entirely dynamically. With Plotly/Dash it's not clear that dynamic
    # layout is possible in this particular way
    
     dbc.Row([
        
        dbc.Col([
            html.Label(children=X_names[0])
        ], width=2),
        
        dbc.Col([
            dcc.Input(id='comm_0',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][0] )
        ], width=2) ,
        
        dbc.Col([
            html.Label(children=X_names[1])
        ], width=2),
         
        dbc.Col([
            dcc.Input(id='comm_1',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][1] )
         ], width=2),
    
        dbc.Col([
            html.Label(children=X_names[2])
        ], width=2),
         
        dbc.Col([
            dcc.Input(id='comm_2',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][2] )
         ], width=2),
     ]),
     
        #row
      dbc.Row([
        
        dbc.Col([
            html.Label(children=X_names[3])
        ], width=2),
        
        dbc.Col([
            dcc.Input(id='comm_3',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][3] )
        ], width=2) ,
        
        dbc.Col([
            html.Label(children=X_names[4])
        ], width=2),
         
        dbc.Col([
            dcc.Input(id='comm_4',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][4] )
         ], width=2),
    
        dbc.Col([
            html.Label(children=X_names[5])
        ], width=2),
         
        dbc.Col([
            dcc.Input(id='comm_5',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][5] )
         ], width=2),
     ]),
     
        #row
       dbc.Row([
        
        dbc.Col([
            html.Label(children=X_names[6])
        ], width=2),
        
        dbc.Col([
            dcc.Input(id='comm_6',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][6] )
        ], width=2) ,
        
        dbc.Col([
            html.Label(children=X_names[7])
        ], width=2),
         
        dbc.Col([
            dcc.Input(id='comm_7',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][7] )
         ], width=2),
    
        dbc.Col([
            html.Label(children=X_names[8])
        ], width=2),
         
        dbc.Col([
            dcc.Input(id='comm_8',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][8] )
         ], width=2),
     ]),
     
        #row
       dbc.Row([
        
        dbc.Col([
            html.Label(children=X_names[9])
        ], width=2),
        
        dbc.Col([
            dcc.Input(id='comm_9',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][9] )
        ], width=2) ,
        
        dbc.Col([
            html.Label(children=X_names[10])
        ], width=2),
         
        dbc.Col([
            dcc.Input(id='comm_10',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][10] )
         ], width=2),
    
        dbc.Col([
            html.Label(children=X_names[11])
        ], width=2),
         
        dbc.Col([
            dcc.Input(id='comm_11',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][11] )
         ], width=2),
     ]),
     
        #row
       dbc.Row([
        
        dbc.Col([
            html.Label(children=X_names[12])
        ], width=2),
        
        dbc.Col([
            dcc.Input(id='comm_12',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][12] )
        ], width=2) ,
        
        dbc.Col([
            html.Label(children=X_names[13])
        ], width=2),
         
        dbc.Col([
            dcc.Input(id='comm_13',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][13] )
         ], width=2),
    
        dbc.Col([
            html.Label(children=X_names[14])
        ], width=2),
         
        dbc.Col([
            dcc.Input(id='comm_14',
            placeholder='Enter commodity price',
            type='number',
            value=X_test[-1][14] )
         ], width=2),
     ]),
     ], style={"border":"1px black solid", "padding": "10px"}),

    html.Div([

        dbc.Row([
            dbc.Col([

                ],width=6),
            ]),
      dbc.Row([ 
        dbc.Col(width=1),  
        dbc.Col([
            html.Button('Submit', id='submit-button'),
            ], width=2),
        dbc.Col([
            html.Div(id='status-div')
            ], width=3),
        ])

    ], style={"padding": "10px"})
   
])
 


# In[24]:


@app.callback(
    Output(component_id='status-div', component_property='children'),
    [Input(component_id='submit-button',  component_property='n_clicks')],
    [State(component_id='comm_0', component_property='value'), State(component_id='comm_1', component_property='value'),
     State(component_id='comm_2', component_property='value'), State(component_id='comm_3', component_property='value'),
     State(component_id='comm_4', component_property='value'), State(component_id='comm_5', component_property='value'),
     State(component_id='comm_6', component_property='value'), State(component_id='comm_7', component_property='value'),
     State(component_id='comm_8', component_property='value'), State(component_id='comm_9', component_property='value'),
     State(component_id='comm_10', component_property='value'), State(component_id='comm_11', component_property='value'),
     State(component_id='comm_12', component_property='value'), State(component_id='comm_13', component_property='value'),
    State(component_id='comm_14', component_property='value'), 
    
    ]    
)
def update_output_div( n_clicks, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14):
    if n_clicks is None:
        raise PreventUpdate
    else:
        
        # First step - put all the c values into a dataframe
        pred_set = pd.DataFrame([[ c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14 ]])
        # In case not all values were entered, carry forward from nearest neighbor (both ways)
        # Real implementation would likely go back and carry forward the most recent value for that particular
        # Region
        
        pred_set[0].fillna(0.0, inplace=True) #Defaulting to zero
        
        print(" Description ", pred_set.describe() )
        print("Head ", pred_set.head() )
        
        prediction = regressor.predict(pred_set)
        pred_val = prediction[0]
        pred_val = '${:.2f}/Ton'.format(pred_val)

        print("Checking out ", prediction)
        return u'''
        Black Sea Milling Wheat price prediction: {}  
        '''.format(pred_val)


# In[ ]:





# In[ ]:


if __name__ == '__main__':
    application.run(debug=False, port=8080)
    #app.run_server(port=8080)


# In[ ]:




