#to do
#clean code
#fix linter errors
#each graph - ai explanation
#sentiment analysis - ai explanation
#wordcloud - ai explanation
#time series - ai explanation
#source breakdown - ai explanation
#candidate sentiment breakdown - ai explanation
#candidate polarity distribution - ai explanation   
#feature to add- csv import export

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random  
from collections import Counter

from sentiment_analyzer import SentimentAnalyzer
from data_generator import DataGenerator
import visualization as viz

app = dash.Dash(__name__, title="m(A)Ind voting: AI Insight into Election Sentiment")
server = app.server  

analyzer = SentimentAnalyzer()

data_generator = DataGenerator()
raw_data = data_generator.generate_dataset(num_samples=500)

analyzed_data = analyzer.analyze_dataframe(raw_data, 'feedback', 'candidate', 'date')

candidate_keywords = {}
for candidate in analyzed_data['candidate'].unique():
    candidate_texts = analyzed_data[analyzed_data['candidate'] == candidate]['feedback'].tolist()
    candidate_keywords[candidate] = analyzer.extract_keywords(candidate_texts, n=10)

global_store = {
    'raw_data': raw_data,
    'analyzed_data': analyzed_data,
    'candidate_keywords': candidate_keywords,
    'using_imported_data': False
}

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1("M(A)Ind Voting: AI Insight into Election Sentiment"),
            html.P("Analyzing opinions on election candidates from surveys and social media")
        ], style={'flex': '1'}),
    ], className='app-header'),
    
    html.Div([
        html.Div([
            dcc.Tabs(id='tabs', value='tab-overview', children=[
                dcc.Tab(label='Overview', value='tab-overview'),
                dcc.Tab(label='Candidate Analysis', value='tab-candidates'),
                dcc.Tab(label='Time Trends', value='tab-time'),
                dcc.Tab(label='Source Analysis', value='tab-source'),
                dcc.Tab(label='Advanced Analysis', value='tab-advanced'),
                dcc.Tab(label='Raw Data', value='tab-data'),
            ])
        ], className='card'),
        
        html.Div(id='tab-content'),
        
        html.Div(id='dummy-div-candidates', style={'display': 'none'}),
        html.Div(id='dummy-div-time', style={'display': 'none'}),
        html.Div(id='dummy-div-source', style={'display': 'none'}),
        html.Div(id='dummy-div-advanced', style={'display': 'none'}),
        html.Div(id='dummy-div-data', style={'display': 'none'}),
        
        html.Div(id='advanced-content', style={'display': 'none'}),
        html.Div(id='sentiment-stats-table', style={'display': 'none'}),
        html.Div(id='candidate-keywords', style={'display': 'none'}),
        html.Div(id='data-table', style={'display': 'none'}),
        
        html.Div(id='source-comparison-overview', style={'display': 'none'}),
        html.Div(id='source-candidate-breakdown', style={'display': 'none'}),
        html.Div(id='source-distribution-text', style={'display': 'none'}),
        html.Div(id='most-positive-source-text', style={'display': 'none'}),
        html.Div(id='most-negative-source-text', style={'display': 'none'}),
        
        html.Div(id='candidate-sentiment-breakdown', style={'display': 'none'}),
        html.Div(id='candidate-polarity-distribution', style={'display': 'none'}),
        html.Div(id='sentiment-time-series', style={'display': 'none'}),
        html.Div(id='candidate-time-series', style={'display': 'none'}),
        html.Div(id='candidate-dropdown', style={'display': 'none'}),

        html.Div([
            dcc.Checklist(id='candidate-checklist', options=[], value=[], style={'display': 'none'}),
            
            dcc.RadioItems(id='time-interval', options=[], value='D', style={'display': 'none'}),
            
            dcc.Dropdown(id='source-candidate-dropdown', options=[], value=None, style={'display': 'none'}),
            
            dcc.Dropdown(id='advanced-viz-dropdown', options=[], value='subjectivity', style={'display': 'none'}),
            
            dcc.Dropdown(id='sentiment-filter', options=[], value='all', style={'display': 'none'})
        ], style={'display': 'none'})
    ], className='dashboard-container')
])

app.clientside_callback(
    """
    function(tab_value) {
        return tab_value;
    }
    """,
    Output('dummy-div-candidates', 'children'),
    Input('tabs', 'value')
)

app.clientside_callback(
    """
    function(tab_value) {
        return tab_value;
    }
    """,
    Output('dummy-div-time', 'children'),
    Input('tabs', 'value')
)

app.clientside_callback(
    """
    function(tab_value) {
        return tab_value;
    }
    """,
    Output('dummy-div-source', 'children'),
    Input('tabs', 'value')
)

app.clientside_callback(
    """
    function(tab_value) {
        return tab_value;
    }
    """,
    Output('dummy-div-advanced', 'children'),
    Input('tabs', 'value')
)

app.clientside_callback(
    """
    function(tab_value) {
        return tab_value;
    }
    """,
    Output('dummy-div-data', 'children'),
    Input('tabs', 'value')
)

def generate_chart_explanation(chart_type, data=None):
    """
    Generate an AI-like explanation for different chart types
    
    Parameters:
    - chart_type: Type of chart (sentiment_breakdown, time_series, keywords, etc.)
    - data: Data specific to the chart (optional)
    
    Returns:
    - HTML component with the explanation
    """
    explanations = {
        'sentiment_breakdown': [
            "This chart shows the distribution of positive, neutral, and negative sentiment across all feedback. The majority sentiment appears to be {}, which suggests that students are generally {} about the election candidates.",
            "The sentiment breakdown reveals that {}% of responses are positive, {}% are neutral, and {}% are negative. This pattern indicates {}.",
            "Looking at this sentiment distribution, we can observe a {} trend. This might indicate that students have {} opinions about the candidates in this election."
        ],
        'time_series': [
            "The sentiment trends over time show {}. This could be related to {} that occurred during this period.",
            "I notice that {} sentiment had a significant {} around {}. This likely correlates with {} during that timeframe.",
            "The time series reveals {} patterns in sentiment. Particularly interesting is the {} which might reflect changing student perspectives as the election campaign progressed."
        ],
        'keywords': [
            "The key topics that emerge from student feedback are centered around {}. This suggests that students are primarily concerned with these issues in this election.",
            "Based on the frequency of keywords, students are most engaged with {} topics. Candidates addressing these concerns might resonate more with the student body.",
            "The keyword analysis shows that {} are prominent topics. Candidates might want to focus on these areas in their campaign messaging to better connect with student priorities."
        ],
        'subjectivity_polarity': [
            "This chart reveals how objective vs. subjective the feedback is, and how positive vs. negative it is. The data points in the {} indicate {}.",
            "The relationship between subjectivity and polarity shows that students tend to express {} when discussing {}. This suggests {}.",
            "Looking at the subjectivity vs. polarity plot, we can see that more subjective comments tend to be {}, while more objective statements are usually {}."
        ],
        'topics': [
            "The topic modeling reveals distinct clusters of related topics. The largest cluster centers around {}, suggesting this is a primary concern for students.",
            "By examining these topic clusters, we can identify that {} and {} are closely related in student discussions, while {} stands as a more isolated concern.",
            "The proximity of topics in this visualization shows how students connect different issues. For example, {} and {} are frequently mentioned together, indicating a perceived relationship between these concerns."
        ],
        'wordcloud': [
            "The word cloud highlights the most frequently used terms in student feedback. The prominence of words like {} suggests these are key themes in election discussions.",
            "From this word cloud, we can see that {} are dominant themes. This gives candidates clear insight into what language resonates with students.",
            "The size of words like {} in this visualization directly correlates to how frequently they appear in student feedback, providing a visual representation of the most discussed topics."
        ]
    }
    
    template = random.choice(explanations.get(chart_type, ["No explanation available for this chart type."]))
    
    if chart_type == 'sentiment_breakdown' and data is not None:
        pos_pct = data['positive_percent'].iloc[0]
        neu_pct = data['neutral_percent'].iloc[0]
        neg_pct = data['negative_percent'].iloc[0]
        
        majority = "positive" if pos_pct > neu_pct and pos_pct > neg_pct else "neutral" if neu_pct > pos_pct and neu_pct > neg_pct else "negative"
        feeling = "optimistic" if majority == "positive" else "undecided" if majority == "neutral" else "concerned"
        
        pattern = "a balanced perspective" if abs(pos_pct - neg_pct) < 20 else "strong opinions" if abs(pos_pct - neg_pct) > 40 else "mixed feelings"
        
        filled_template = template.format(
            majority, feeling,
            round(pos_pct, 1), round(neu_pct, 1), round(neg_pct, 1), pattern,
            "balanced" if abs(pos_pct - neg_pct) < 20 else "polarized" if abs(pos_pct - neg_pct) > 40 else "mixed",
            "diverse" if neu_pct > 30 else "strong"
        )
    else:
        filled_template = template.format(
            "multiple peaks and valleys", "campaign events or policy announcements",
            "positive", "spike", "mid-campaign", "major policy announcements",
            "cyclical", "convergence of sentiment near the end",
            "education, sustainability, and campus life", 
            "policy, affordability, and campus improvement",
            "academic resources, campus facilities, and student support services",
            "upper right quadrant", "highly subjective positive feedback",
            "more emotional when discussing personal impacts", "policy details", "a personal connection to these issues",
            "very positive", "more neutral",
            "education policy", 
            "housing affordability", "campus life", "environmental concerns",
            "academic resources", "student support services",
            "policy, education, and campus", 
            "sustainability, affordability, and leadership", 
            "leadership, policy, and vision"
        )
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-robot", style={"marginRight": "10px", "fontSize": "18px"}),
            html.Span("AI Insight", style={"fontWeight": "bold"})
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px", "color": "#6366f1"}),
        html.P(filled_template, style={"color": "var(--text-color)"})
    ], className="ai-explanation-box")    

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-overview':
        try:
            fig_overall = viz.create_sentiment_breakdown(global_store['analyzed_data'])
            fig_overall.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=50, b=30),
                title="Sentiment Breakdown by Candidate",
                title_x=0.5,
                font=dict(family="Inter, sans-serif"),
                colorway=["#10b981", "#3b82f6", "#ef4444"] 
            )
            
            fig_time = viz.create_sentiment_time_series(global_store['analyzed_data'], group_by=None)
            fig_time.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=50, b=30),
                title="Sentiment Trends Over Time",
                title_x=0.5,
                font=dict(family="Inter, sans-serif"),
                colorway=["#10b981", "#3b82f6", "#ef4444"] 
            )
            
            all_keywords = analyzer.extract_keywords(global_store['analyzed_data']['feedback'].tolist(), n=15)
            
            if isinstance(all_keywords, list):
                all_keywords = dict(all_keywords)
                
            fig_keywords = viz.create_keyword_chart(all_keywords, display_count=15)
            fig_keywords.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=50, b=30),
                title="Top Keywords in Feedback",
                title_x=0.5,
                font=dict(family="Inter, sans-serif")
            )
            
            stats = analyzer.get_sentiment_statistics(global_store['analyzed_data'])
            
            sentiment_explanation = generate_chart_explanation('sentiment_breakdown', stats)
            time_explanation = generate_chart_explanation('time_series')
            keyword_explanation = generate_chart_explanation('keywords')
            
            summary_cards = html.Div([
                html.Div([
                    html.Div([
                        html.H3("Positive Sentiment", style={'textAlign': 'center', 'color': '#10b981', 'margin': '0 0 5px 0'}),
                        html.Div(f"{stats['positive_percent'].iloc[0]:.1f}%", 
                            style={'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center'})
                    ], className='summary-card', style={'borderTop': '3px solid #10b981'}),
                    
                    html.Div([
                        html.H3("Neutral Sentiment", style={'textAlign': 'center', 'color': '#3b82f6', 'margin': '0 0 5px 0'}),
                        html.Div(f"{stats['neutral_percent'].iloc[0]:.1f}%", 
                            style={'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center'})
                    ], className='summary-card', style={'borderTop': '3px solid #3b82f6'}),
                    
                    html.Div([
                        html.H3("Negative Sentiment", style={'textAlign': 'center', 'color': '#ef4444', 'margin': '0 0 5px 0'}),
                        html.Div(f"{stats['negative_percent'].iloc[0]:.1f}%", 
                            style={'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center'})
                    ], className='summary-card', style={'borderTop': '3px solid #ef4444'}),
                    
                    html.Div([
                        html.H3("Total Feedback", style={'textAlign': 'center', 'color': '#e6e6e6', 'margin': '0 0 5px 0'}),
                        html.Div(f"{stats['count'].iloc[0]}", 
                            style={'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center'})
                    ], className='summary-card', style={'borderTop': '3px solid #e6e6e6'})
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '15px', 'justifyContent': 'space-between', 'marginBottom': '20px'})
            ])
            
            return html.Div([
                summary_cards,
                
                html.Div([
                    html.H2("Overall Election Sentiment", className='card-title'),
                    
                    html.Div([
                        html.Div([
                            dcc.Graph(figure=fig_overall, config={'displayModeBar': False})
                        ], className='graph-container', style={'width': '50%', 'height': '100%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                        
                        html.Div([
                            dcc.Graph(figure=fig_time, config={'displayModeBar': False})
                        ], className='graph-container', style={'width': '50%', 'height': '100%', 'display': 'inline-block', 'verticalAlign': 'top'})
                    ], style={'display': 'flex', 'alignItems': 'stretch', 'gap': '0px'}),
                    
                    html.Div([
                        html.Div([
                            sentiment_explanation
                        ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px', 'boxSizing': 'border-box'}),
                        
                        html.Div([
                            time_explanation
                        ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px', 'boxSizing': 'border-box'})
                    ], style={'display': 'flex', 'flexWrap': 'wrap'})
                ], className='card'),
                
                html.Div([
                    html.H2("Key Topics in Student Feedback", className='card-title'),
                    dcc.Graph(figure=fig_keywords, config={'displayModeBar': False}),
                    
                    keyword_explanation
                ], className='card')
            ])
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in overview tab: {str(e)}\n{error_details}")
            
            return html.Div([
                html.Div([
                    html.H2("Overview", className='card-title'),
                    html.Div([
                        html.Div([
                            html.H3("Error Loading Overview", 
                                   style={'color': '#ff6b6b', 'textAlign': 'center', 'margin': '20px 0'}),
                            html.P(f"We encountered an issue while loading the overview: {str(e)}", 
                                  style={'textAlign': 'center', 'marginBottom': '15px'}),
                            html.P("Please try refreshing the page or select a different tab to continue.",
                                  style={'textAlign': 'center'})
                        ], style={'padding': '20px', 'backgroundColor': 'rgba(255, 107, 107, 0.1)', 
                                 'borderRadius': '8px', 'margin': '20px auto', 'maxWidth': '800px'})
                    ])
                ], className='card')
            ])
    
    elif tab == 'tab-candidates':
        return html.Div([
            html.Div([
                html.H2("Candidate Sentiment Analysis", className='card-title'),
                
                html.Div([
                    html.Label("Select Candidates:"),
                    dcc.Checklist(
                        id='candidate-checklist',
                        options=[{'label': c, 'value': c} for c in global_store['analyzed_data']['candidate'].unique()],
                        value=global_store['analyzed_data']['candidate'].unique().tolist(),
                        className='candidate-checklist',
                        inputClassName='checklist-input',
                        labelClassName='checklist-label'
                    )
                ], className='control-container'),
                
                html.Div(dcc.Graph(id='candidate-sentiment-breakdown', config={'displayModeBar': False})),
                html.Div(dcc.Graph(id='candidate-polarity-distribution', config={'displayModeBar': False}))
            ], className='card'),
            
            html.Div([
                html.H2("Candidate Keywords", className='card-title'),
                html.Div(id='candidate-keywords', className='keywords-container')
            ], className='card')
        ])
    
    elif tab == 'tab-time':
        return html.Div([
            html.Div([
                html.H2("Sentiment Over Time", className='card-title'),
                
                html.Div([
                    html.Label("Time Interval:"),
                    dcc.RadioItems(
                        id='time-interval',
                        options=[
                            {'label': 'Daily', 'value': 'D'},
                            {'label': 'Weekly', 'value': 'W'},
                            {'label': 'Monthly', 'value': 'M'}
                        ],
                        value='D',
                        className='time-interval-radio',
                        inputClassName='radio-input',
                        labelClassName='radio-label'
                    )
                ], className='control-container'),
                
                html.Div(dcc.Graph(id='sentiment-time-series'))
            ], className='card'),
            
            html.Div([
                html.H2("Candidate Sentiment Over Time", className='card-title'),
                html.Div(dcc.Graph(id='candidate-time-series'))
            ], className='card')
        ])
    
    elif tab == 'tab-source':
        return html.Div([
            html.Div([
                html.H2("Source Comparison Overview", className='card-title'),
                html.P("This visualization shows how sentiment is distributed across different feedback sources.", className="card-description"),
                html.Div(dcc.Graph(id='source-comparison-overview'))
            ], className='card'),
            
            html.Div([
                html.H2("Source Breakdown by Candidate", className='card-title'),
                html.P("See how sentiment differs across sources for each candidate.", className="card-description"),
                
                html.Div([
                    html.Label("Select Candidate:"),
                    dcc.Dropdown(
                        id='source-candidate-dropdown',
                        options=[{'label': c, 'value': c} for c in global_store['analyzed_data']['candidate'].unique()],
                        value=global_store['analyzed_data']['candidate'].unique()[0],
                        className='candidate-dropdown'
                    )
                ], className='control-container'),
                
                html.Div(dcc.Graph(id='source-candidate-breakdown'))
            ], className='card'),
            
            html.Div([
                html.H2("Source Analysis Insights", className='card-title'),
                html.P("Key insights about how different sources contribute to the overall sentiment analysis:", className="card-description"),
                html.Div([
                    html.Div([
                        html.H3("Source Distribution", className="insight-title"),
                        html.P(id="source-distribution-text", className="insight-text")
                    ], className="insight-card"),
                    html.Div([
                        html.H3("Most Positive Source", className="insight-title"),
                        html.P(id="most-positive-source-text", className="insight-text")
                    ], className="insight-card"),
                    html.Div([
                        html.H3("Most Negative Source", className="insight-title"),
                        html.P(id="most-negative-source-text", className="insight-text")
                    ], className="insight-card")
                ], className="insights-container")
            ], className='card')
        ])
    
    elif tab == 'tab-advanced':
        advanced_tab = html.Div([
            html.Div([
                html.Div([
                    html.Label("Select Candidate:", className="input-label"),
                    html.Div(dcc.Dropdown(
                        id='candidate-dropdown',
                        options=[],
                        value=None,
                        clearable=True,
                        searchable=True,
                        placeholder="Select a candidate or view all",
                        className='dropdown'
                    ))
                ], className="filter-group")
            ], className="filters-container"),
            html.Div(id='advanced-content', className='tab-content')
        ], id='tab-advanced', className='tab-content')
        
        return advanced_tab
    
    elif tab == 'tab-data':
        return html.Div([
            html.Div([
                html.H2("Raw Data with Sentiment Analysis", className='card-title'),
                
                html.Div([
                    html.Label("Filter by Sentiment:"),
                    dcc.Dropdown(
                        id='sentiment-filter',
                        options=[
                            {'label': 'All', 'value': 'all'},
                            {'label': 'Positive', 'value': 'positive'},
                            {'label': 'Neutral', 'value': 'neutral'},
                            {'label': 'Negative', 'value': 'negative'}
                        ],
                        value='all',
                        className='sentiment-filter-dropdown'
                    )
                ], className='control-container'),
                
                html.Div(id='data-table', className='data-table-container')
            ], className='card')
        ])

@app.callback(
    Output('candidate-sentiment-breakdown', 'figure'),
    [Input('candidate-checklist', 'value'),
     Input('dummy-div-candidates', 'children')]
)
def update_candidate_breakdown(selected_candidates, tab_value):
    if tab_value != 'tab-candidates':
        return go.Figure()
        
    filtered_data = global_store['analyzed_data'][global_store['analyzed_data']['candidate'].isin(selected_candidates)]
    fig = viz.create_sentiment_breakdown(filtered_data)
    return fig

@app.callback(
    Output('candidate-polarity-distribution', 'figure'),
    [Input('candidate-checklist', 'value'),
     Input('dummy-div-candidates', 'children')]
)
def update_polarity_distribution(selected_candidates, tab_value):
    if tab_value != 'tab-candidates':
        return go.Figure()
        
    filtered_data = global_store['analyzed_data'][global_store['analyzed_data']['candidate'].isin(selected_candidates)]
    fig = viz.create_polarity_distribution(filtered_data)
    return fig

@app.callback(
    Output('candidate-keywords', 'children'),
    [Input('candidate-checklist', 'value'),
     Input('dummy-div-candidates', 'children')]
)
def update_candidate_keywords(selected_candidates, tab_value):
    if tab_value != 'tab-candidates':
        return html.Div()
        
    keyword_graphs = []
    
    for i, candidate in enumerate(selected_candidates):
        if candidate in global_store['candidate_keywords']:
            fig = viz.create_keyword_chart(
                global_store['candidate_keywords'][candidate], 
                display_count=10
            )
            
            fig.update_layout(
                margin=dict(l=20, r=20, t=50, b=30),
                height=400,
                title={
                    'text': f'Top Keywords for {candidate}',
                    'y':0.97,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 16, 'family': 'Inter, sans-serif'}
                },
                font=dict(family="Inter, sans-serif")
            )
            
            keyword_graphs.append(
                html.Div([
                    dcc.Graph(figure=fig)
                ], style={
                    'width': '50%', 
                    'display': 'inline-block',
                    'padding': '10px',
                    'box-sizing': 'border-box'
                })
            )
    
    return html.Div(keyword_graphs, style={'display': 'flex', 'flex-wrap': 'wrap'})

@app.callback(
    Output('sentiment-time-series', 'figure'),
    [Input('time-interval', 'value'),
     Input('dummy-div-time', 'children')]
)
def update_sentiment_time_series(time_interval, tab_value):
    if tab_value != 'tab-time':
        return go.Figure()
        
    fig = viz.create_sentiment_time_series(global_store['analyzed_data'], group_by=None, time_interval=time_interval)
    return fig

@app.callback(
    Output('candidate-time-series', 'figure'),
    [Input('time-interval', 'value'),
     Input('dummy-div-time', 'children')]
)
def update_candidate_time_series(time_interval, tab_value):
    if tab_value != 'tab-time':
        return go.Figure()
        
    fig = viz.create_sentiment_time_series(global_store['analyzed_data'], time_interval=time_interval)
    return fig

@app.callback(
    Output('source-candidate-breakdown', 'figure'),
    [Input('source-candidate-dropdown', 'value'),
     Input('dummy-div-source', 'children')]
)


def update_source_candidate_breakdown(selected_candidate, tab_value):
    if tab_value != 'tab-source':
        return go.Figure()
    
    try:
        if 'analyzed_data' not in global_store or global_store['analyzed_data'] is None or global_store['analyzed_data'].empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for visualization",
                showarrow=False,
                font=dict(size=16),
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )
            return fig
        
        if selected_candidate is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Please select a candidate",
                showarrow=False,
                font=dict(size=16),
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )
            return fig
            
        filtered_data = global_store['analyzed_data'][global_store['analyzed_data']['candidate'] == selected_candidate]
        if filtered_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {selected_candidate}",
                showarrow=False,
                font=dict(size=16),
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )
            return fig
            
        fig = viz.create_sentiment_breakdown(filtered_data, group_by='source')
        return fig
    except Exception as e:
        print(f"Error updating source candidate breakdown: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            showarrow=False,
            font=dict(size=14),
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        return fig

@app.callback(
    Output('data-table', 'children'),
    [Input('sentiment-filter', 'value'),
     Input('dummy-div-data', 'children')]
)
def update_data_table(sentiment_filter, tab_value):
    if tab_value != 'tab-data':
        return html.Div()
        
    if sentiment_filter == 'all':
        filtered_df = global_store['analyzed_data']
    else:
        filtered_df = global_store['analyzed_data'][global_store['analyzed_data']['sentiment'] == sentiment_filter]
    
    data_sample = filtered_df.sort_values('date')
    
    data_rows = []
    for _, row in data_sample.iterrows():
        if row['sentiment'] == 'positive':
            badge_color = '#10b981' 
        elif row['sentiment'] == 'negative':
            badge_color = '#ef4444'  # Red
        else:
            badge_color = '#3b82f6' 
            
        data_rows.append(html.Tr([
            html.Td(row['date'].strftime('%Y-%m-%d')),
            html.Td(row['candidate'], style={'font-weight': '500'}),
            html.Td(row['feedback']),
            html.Td(html.Div(row['source'], style={
                'display': 'inline-block',
                'padding': '2px 8px',
                'border-radius': '4px',
                'background-color': '#1e1e30',
                'font-size': '12px'
            })),
            html.Td([
                html.Div(style={
                    'display': 'flex',
                    'align-items': 'center'
                }, children=[
                    html.Div(style={
                        'width': '10px',
                        'height': '10px',
                        'borderRadius': '50%',
                        'backgroundColor': badge_color,
                        'marginRight': '6px'
                    }),
                    html.Span(f"{row['polarity']:.3f}",
                        style={'font-weight': 'bold', 
                                'color': badge_color})
                ])
            ]),
            html.Td([
                html.Span(row['sentiment'], style={
                    'background-color': badge_color,
                    'color': '#252538',
                    'padding': '3px 8px',
                    'border-radius': '12px',
                    'font-size': '12px',
                    'font-weight': '600'
                })
            ])
        ], style={'background-color': f"rgba({int(badge_color[1:3], 16)}, {int(badge_color[3:5], 16)}, {int(badge_color[5:7], 16)}, 0.05)"}))
    
    data_table = html.Table([
        html.Thead(
            html.Tr([
                html.Th('Date'),
                html.Th('Candidate'),
                html.Th('Feedback'),
                html.Th('Source'),
                html.Th('Polarity'),
                html.Th('Sentiment')
            ])
        ),
        html.Tbody(data_rows)
    ], style={'width': '100%', 'border-collapse': 'collapse'})
    
    return data_table

@app.callback(
    Output('advanced-content', 'children'),
    [Input('candidate-dropdown', 'value')],
    prevent_initial_call=False
)
def update_advanced_tab(selected_candidate):
    """
    Updates the advanced tab content based on the selected candidate.
    
    Args:
        selected_candidate: The selected candidate from the dropdown
    
    Returns:
        Component with visualizations and statistics
    """
    try:
        global_data = global_store.get('analyzed_data')
        if global_data is None or global_data.empty:
            return html.Div([
                html.H3("No Data Available", className="error-title"),
                html.P("Please upload and analyze data first.", className="error-message"),
                html.Div(html.I(className="fas fa-exclamation-circle"), className="error-icon")
            ], className="error-container")
        
        scatter_fig = create_subjectivity_polarity_chart(global_data, selected_candidate)
        stats_components = generate_sentiment_stats(global_data, selected_candidate)
        
        return html.Div([
            html.Div([
                html.H2("Advanced Sentiment Analysis", className="section-title"),
                html.P("Explore the relationship between subjectivity and polarity in sentiments and view detailed statistics.", 
                       className="section-description"),
                
                html.Div([
                    dcc.Graph(
                        figure=scatter_fig,
                        className="sentiment-scatter"
                    )
                ], className="chart-container"),
                
                html.Div(stats_components, className="stats-container")
            ], className="advanced-content-container")
        ])
        
    except Exception as e:
        print(f"Error updating advanced tab: {e}")
        return html.Div([
            html.H3("Error Loading Data", className="error-title"),
            html.P(f"An error occurred: {str(e)}", className="error-message"),
            html.P("Please try refreshing the page or selecting a different candidate.", className="error-advice"),
            html.Div(html.I(className="fas fa-exclamation-circle"), className="error-icon")
        ], className="error-container")

def create_subjectivity_polarity_chart(df, selected_candidate=None):
    """
    Creates a scatter plot of subjectivity vs polarity with quadrant annotations.
    
    Args:
        df: DataFrame with the sentiment data
        selected_candidate: Optional candidate to filter by
    
    Returns:
        Plotly figure object
    """
    try:
        if df is None or df.empty:
            return go.Figure().update_layout(
                title="No sentiment data available",
                annotations=[dict(
                    text="No data to display. Please import data first.",
                    showarrow=False,
                    font=dict(size=16),
                    xref="paper", yref="paper",
                    x=0.5, y=0.5
                )]
            )
        
        if selected_candidate and selected_candidate in df['candidate'].unique():
            chart_df = df[df['candidate'] == selected_candidate].copy()
        else:
            chart_df = df.copy()
        



        fig = px.scatter(
            chart_df, 
            x='polarity', 
            y='subjectivity',
            color='candidate',
            hover_data=['feedback', 'date'],
            title=f"Subjectivity vs Polarity {'for ' + selected_candidate if selected_candidate else 'for All Candidates'}",
            template="plotly_dark"
        )
        
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="#aaaaaa"),
            x0=-1, y0=0.5, x1=1, y1=0.5
        )
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="#aaaaaa"),
            x0=0, y0=0, x1=0, y1=1
        )
        


        fig.add_annotation(x=0.5, y=0.75, text="Positive Subjective", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=-0.5, y=0.75, text="Negative Subjective", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=0.5, y=0.25, text="Positive Objective", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=-0.5, y=0.25, text="Negative Objective", showarrow=False, font=dict(size=12))
        
        fig.update_layout(
            xaxis=dict(title="Polarity", range=[-1.1, 1.1]),
            yaxis=dict(title="Subjectivity", range=[-0.1, 1.1]),
            plot_bgcolor="#1e293b",
            paper_bgcolor="#1e293b",
            font=dict(family="Inter, sans-serif", color="#f1f5f9"),
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    except Exception as e:
        print(f"Error creating subjectivity polarity chart: {e}")
        return go.Figure().update_layout(
            title="Error creating chart",
            annotations=[dict(
                text=f"An error occurred: {str(e)}",
                showarrow=False,
                font=dict(size=14),
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )

def generate_sentiment_stats(df, selected_candidate=None):
    """
    Generates sentiment statistics for a selected candidate or all candidates.
    
    Args:
        df: DataFrame with the sentiment data
        selected_candidate: Optional candidate to filter by
    
    Returns:
        List of statistics components
    """
    try:
        if df is None or df.empty:
            return [html.Div("No data available for statistics.")]
        
        if selected_candidate and selected_candidate in df['candidate'].unique():
            stats_df = df[df['candidate'] == selected_candidate].copy()
            title = f"Sentiment Statistics for {selected_candidate}"
        else:
            stats_df = df.copy()
            title = "Overall Sentiment Statistics"
        
        avg_polarity = stats_df['polarity'].mean()
        avg_subjectivity = stats_df['subjectivity'].mean()
        
        positive_pct = (stats_df['polarity'] > 0.05).mean() * 100
        negative_pct = (stats_df['polarity'] < -0.05).mean() * 100
        neutral_pct = ((stats_df['polarity'] >= -0.05) & (stats_df['polarity'] <= 0.05)).mean() * 100
        
        subjective_pct = (stats_df['subjectivity'] > 0.5).mean() * 100
        objective_pct = (stats_df['subjectivity'] <= 0.5).mean() * 100
        
        total_comments = len(stats_df)
        unique_users = stats_df['date'].nunique() if 'date' in stats_df.columns else "N/A"
        
        stats = [
            html.H3(title, className="stats-title"),
            
            html.Div([
                html.Div([
                    html.H4("Average Sentiment", className="stat-card-title"),
                    html.Div([
                        html.Div(f"{avg_polarity:.2f}", className="stat-value"),
                        html.Div("Polarity", className="stat-label")
                    ], className="stat-item"),
                    html.Div([
                        html.Div(f"{avg_subjectivity:.2f}", className="stat-value"),
                        html.Div("Subjectivity", className="stat-label")
                    ], className="stat-item")
                ], className="stat-card"),
                
                html.Div([
                    html.H4("Sentiment Distribution", className="stat-card-title"),
                    html.Div([
                        html.Div(f"{positive_pct:.1f}%", className="stat-value positive"),
                        html.Div("Positive", className="stat-label")
                    ], className="stat-item"),
                    html.Div([
                        html.Div(f"{neutral_pct:.1f}%", className="stat-value neutral"),
                        html.Div("Neutral", className="stat-label")
                    ], className="stat-item"),
                    html.Div([
                        html.Div(f"{negative_pct:.1f}%", className="stat-value negative"),
                        html.Div("Negative", className="stat-label")
                    ], className="stat-item")
                ], className="stat-card"),
                
                html.Div([
                    html.H4("Subjectivity Distribution", className="stat-card-title"),
                    html.Div([
                        html.Div(f"{subjective_pct:.1f}%", className="stat-value"),
                        html.Div("Subjective", className="stat-label")
                    ], className="stat-item"),
                    html.Div([
                        html.Div(f"{objective_pct:.1f}%", className="stat-value"),
                        html.Div("Objective", className="stat-label")
                    ], className="stat-item")
                ], className="stat-card"),
                
                html.Div([
                    html.H4("Engagement", className="stat-card-title"),
                    html.Div([
                        html.Div(str(total_comments), className="stat-value"),
                        html.Div("Total Comments", className="stat-label")
                    ], className="stat-item"),
                    html.Div([
                        html.Div(str(unique_users), className="stat-value"),
                        html.Div("Unique Users", className="stat-label")
                    ], className="stat-item")
                ], className="stat-card")
            ], className="stats-grid")
        ]
        
        return stats
    
    except Exception as e:
        print(f"Error generating sentiment stats: {e}")
        return [html.Div(f"Error generating statistics: {str(e)}")]

@app.callback(
    Output('candidate-dropdown', 'options'),
    [Input('dummy-div-advanced', 'children')]
)
def update_candidate_dropdown(tab_value):
    if tab_value != 'tab-advanced':
        return []
        
    try:
        candidates = global_store['analyzed_data']['candidate'].unique()
        return [{'label': candidate, 'value': candidate} for candidate in candidates]
    except Exception as e:
        print(f"Error updating candidate dropdown: {str(e)}")
        return []

@app.callback(
    Output('source-comparison-overview', 'figure'),
    [Input('dummy-div-source', 'children')]
)
def update_source_comparison_overview(tab_value):
    if tab_value != 'tab-source':
        return go.Figure()
    
    try:
        if 'analyzed_data' not in global_store or global_store['analyzed_data'] is None or global_store['analyzed_data'].empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for visualization",
                showarrow=False,
                font=dict(size=16),
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )
            return fig
            
        fig = viz.create_source_comparison(global_store['analyzed_data'])
        return fig
    except Exception as e:
        print(f"Error updating source comparison: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            showarrow=False,
            font=dict(size=14),
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        return fig

@app.callback(
    [Output('source-distribution-text', 'children'),
     Output('most-positive-source-text', 'children'),
     Output('most-negative-source-text', 'children')],
    [Input('dummy-div-source', 'children')]
)
def update_source_insights(tab_value):
    if tab_value != 'tab-source':
        return "", "", ""
    
    try:
        if 'analyzed_data' not in global_store or global_store['analyzed_data'] is None or global_store['analyzed_data'].empty:
            return "No data available", "No data available", "No data available"
        source_counts = global_store['analyzed_data']['source'].value_counts()
        total_records = len(global_store['analyzed_data'])
        
        source_distribution = ", ".join([
            f"{source}: {count} records ({count/total_records*100:.1f}%)" 
            for source, count in source_counts.items()
        ])
        
        if 'source' not in global_store['analyzed_data'].columns:
            return "Source data not available", "Source data not available", "Source data not available"
            
        source_sentiment = global_store['analyzed_data'].groupby('source')['polarity'].agg(['mean', 'count']).reset_index()
        
        if source_sentiment.empty:
            return source_distribution, "No sentiment data available", "No sentiment data available"
            
        source_sentiment = source_sentiment.sort_values('mean', ascending=False)
        reliable_sources = source_sentiment[source_sentiment['count'] >= 5]
        
        if reliable_sources.empty:
            most_positive = "Insufficient data (need at least 5 records per source)"
            most_negative = "Insufficient data (need at least 5 records per source)"
        else:
            most_positive_source = reliable_sources.iloc[0]
            most_negative_source = reliable_sources.iloc[-1]
            
            most_positive = f"{most_positive_source['source']} with average polarity of {most_positive_source['mean']:.3f} across {most_positive_source['count']} records"
            most_negative = f"{most_negative_source['source']} with average polarity of {most_negative_source['mean']:.3f} across {most_negative_source['count']} records"
        
        return source_distribution, most_positive, most_negative
    
    except Exception as e:
        print(f"Error in source insights: {e}")
        error_msg = f"Error processing data: {str(e)}"
        return error_msg, error_msg, error_msg

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>MindThePolls - Student Election Sentiment Analysis</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            :root {
                --primary-color: #232353;
                --primary-hover: #6361a4;

                --positive-color: #10b981;
                --neutral-color: #3b82f6;
                --negative-color: #ef4444;
                --background-color: #111827;
                --card-bg: #1e1e2e;
                --text-color: #f1f5f9;
                --subtext-color: #94a3b8;
                --border-color: #2d3748;
            }
            
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--background-color);
                color: var(--text-color);
                line-height: 1.6;
                margin: 0;
                padding: 0;
                transition: all 0.3s ease;
            }
            
            h1, h2, h3, h4, h5, h6 {
                font-weight: 600;
                margin-top: 0;
                color: var(--text-color);
            }
            
            h1 { font-size: 2rem; margin-bottom: 1rem; }
            h2 { font-size: 1.5rem; margin-bottom: 0.75rem; }
            h3 { font-size: 1.25rem; margin-bottom: 0.5rem; }
            
            p {
                margin-top: 0;
                margin-bottom: 1rem;
            }
            
            a {
                color: var(--primary-color);
                text-decoration: none;
                transition: color 0.2s;
            }
            
            a:hover {
                color: var(--primary-hover);
            }
            
            button, .button {
                background-color: var(--primary-color);
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 0.375rem;
                cursor: pointer;
                font-size: 0.875rem;
                font-weight: 500;
                transition: background-color 0.2s;
            }
            
            button:hover, .button:hover {
                background-color: var(--primary-hover);
            }
            
            /* Dashboard CONTAINER*/
            .dashboard-container {
                max-width: 1280px;
                margin: 0 auto;
                padding: 1rem;
            }
            
            /* CARDS */
            .card {
                background-color: var(--card-bg);
                border-radius: 0.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
            }
            
            .card-title {
                font-size: 1.25rem;
                font-weight: 600;
                margin-top: 0;
                margin-bottom: 1rem;
                color: var(--text-color);
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 0.75rem;
            }
            
            .card-description {
                color: var(--subtext-color);
                font-size: 0.95rem;
                margin-bottom: 1.25rem;
            }
            
            /*APP HEADER */

            .app-header {
            background: linear-gradient(90deg, var(--primary-color), var(--primary-hover));
            padding: 2.5rem;
            margin-bottom: 2rem;
            border-radius: 0 0 1.5rem 1.5rem;
            box-shadow: var(--shadow-medium);
            }
            
            .app-title {
                margin: 0;
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--text-color);
                display: flex;
                align-items: center;
            }
            
            .app-title .fa-chart-line {
                margin-right: 0.5rem;
                color: var(--primary-color);
            }
            
            .app-controls {
                display: flex;
                gap: 0.75rem;
            }
            
            /*Tabs*/
            .tabs {
                margin-bottom: 1.5rem;
                color: var(--primary-color);

            }
            .tab {
                padding: 0.75rem 1.25rem;
                font-size: 0.875rem;
                font-weight: 500;
                color: #334155;
                border-bottom: 2px solid transparent;
                transition: all 0.2s ease;
            }
            
            .tab--selected {
                color: var(--primary-color);
                border-bottom-color: var(--primary-color);
                font-weight: 600;
            }
            
            /*Graphs and Visualizations*/
            .js-plotly-plot {
                border-radius: 0.375rem;
                overflow: hidden;
            }
            
            /*     Interactive Elements */
            .checklist-input, .radio-input {
                margin-right: 0.5rem;
            }
            
            .checklist-label, .radio-label {
                margin-right: 1rem;
                color: var(--text-color);
            }
            
            .control-container {
                margin-bottom: 1.25rem;
                padding: 1rem;
                background-color: rgba(0, 0, 0, 0.2);
                border-radius: 0.375rem;
            }
            
            .Select-control {
                background-color: rgba(255, 255, 255, 0.05) !important;
                border-color: var(--border-color) !important;
                color: var(--text-color) !important;
                border-radius: 0.375rem !important;
            }
            
            .Select-menu-outer {
                background-color: var(--card-bg) !important;
                border-color: var(--border-color) !important;
                color: var(--text-color) !important;
                border-radius: 0 0 0.375rem 0.375rem !important;
                z-index: 10 !important;
            }
            
            .Select-option {
                background-color: var(--card-bg) !important;
                color: var(--text-color) !important;
            }
            
            .Select-option.is-focused {
                background-color: rgba(99, 102, 241, 0.1) !important;
            }
            
            .Select-option.is-selected {
                background-color: var(--primary-color) !important;
            }
            
            .Select-value-label {
                color: var(--text-color) !important;
            }
            
            .Select-placeholder, .Select--single > .Select-control .Select-value {
                color: var(--subtext-color) !important;
            }
            
            /* Data table styling */
            table {
                width: 100%;
                border-collapse: collapse;
                color: var(--text-color);
                font-size: 0.875rem;
            }
            
            th {
                text-align: left;
                padding: 0.75rem 1rem;
                background-color: rgba(0, 0, 0, 0.2);
                font-weight: 500;
                color: var(--subtext-color);
                border-bottom: 1px solid var(--border-color);
            }
            
            td {
                padding: 0.75rem 1rem;
                border-bottom: 1px solid var(--border-color);
                vertical-align: middle;
            }
            
            tr:last-child td {
                border-bottom: none;
            }
            
            tr:hover {
                background-color: rgba(255, 255, 255, 0.05);
            }
            
            /* SummarY CARDS*/
            .summary-cards {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 1rem;
                margin-bottom: 1.5rem;
            }
            
            .summary-card {
                padding: 1.25rem;
                background-color: rgba(0, 0, 0, 0.2);
                border-radius: 0.375rem;
                text-align: center;
                transition: all 0.2s ease;
            }
            
            .summary-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .summary-value {
                font-size: 1.75rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                color: var(--primary-color);
            }
            
            .summary-label {
                font-size: 0.875rem;
                color: var(--subtext-color);
                font-weight: 500;
            }
            
            /*animation */
            ._dash-loading {
                margin: auto;
                color: var(--primary-color);
                width: 100%;
                height: 100%;
                top: 0;
                left: 0;
                position: fixed;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: rgba(0, 0, 0, 0.4);
                z-index: 1000;
                backdrop-filter: blur(5px);
            }
            
            ._dash-loading::after {
                content: "";
                display: block;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                border: 3px solid var(--primary-color);
                border-color: var(--primary-color) transparent var(--primary-color) transparent;
                animation: loading-ring 1.2s linear infinite;
            }
            
            /* Error  */
            .error-container {
                background-color: rgba(239, 68, 68, 0.1);
                border-left: 4px solid var(--negative-color);
                padding: 1.5rem;
                border-radius: 0.375rem;
                margin: 1rem 0;
            }
            
            .error-title {
                color: var(--negative-color);
                margin-top: 0;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .error-message, .error-advice {
                color: var(--text-color);
                margin-bottom: 0.5rem;
            }
            
            .error-icon {
                color: var(--negative-color);
                font-size: 1.5rem;
                margin-bottom: 1rem;
            }
            
            /* Insights */
            .insights-container {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 1.25rem;
                margin-top: 1.5rem;
            }
            
            .insight-card {
                background-color: rgba(0, 0, 0, 0.15);
                border-radius: 0.375rem;
                padding: 1.25rem;
                border-left: 3px solid var(--primary-color);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .insight-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .insight-title {
                color: var(--text-color);
                font-size: 1.1rem;
                margin-top: 0;
                margin-bottom: 0.75rem;
                font-weight: 600;
            }
            
            .insight-text {
                color: var(--subtext-color);
                font-size: 0.95rem;
                margin-bottom: 0;
                line-height: 1.5;
            }

            /* Advancedtab */
            .advanced-content-container {
                padding: 1.5rem;
                background-color: rgba(0, 0, 0, 0.05);
                border-radius: 0.5rem;
            }
            
            .section-title {
                font-size: 1.75rem;
                font-weight: 600;
                margin-bottom: 0.75rem;
                color: var(--text-color);
            }
            
            .section-description {
                color: var(--subtext-color);
                margin-bottom: 1.5rem;
                font-size: 1rem;
                line-height: 1.5;
            }
            
            .chart-container {
                background-color: rgba(0, 0, 0, 0.1);
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .sentiment-scatter {
                min-height: 500px;
            }
            
            .stats-container {
                margin-top: 2rem;
            }
            
            .stats-title {
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                text-align: center;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 1.25rem;
            }
            
            .stat-card {
                background-color: rgba(0, 0, 0, 0.1);
                border-radius: 0.5rem;
                padding: 1.25rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }
            
            .stat-card-title {
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 1rem;
                text-align: center;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 0.5rem;
            }
            
            .stat-item {
                margin-bottom: 0.75rem;
                text-align: center;
            }
            
            .stat-value {
                font-size: 1.5rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }
            
            .stat-value.positive {
                color: var(--positive-color);
            }
            
            .stat-value.neutral {
                color: var(--neutral-color);
            }
            
            .stat-value.negative {
                color: var(--negative-color);
            }
            
            .stat-label {
                font-size: 0.9rem;
                color: var(--subtext-color);
            }
            
            /* AI Explanation Box */
            .ai-explanation-box {
                background-color: rgba(99, 102, 241, 0.1);
                border-left: 4px solid var(--primary-color);
                padding: 1rem;
                margin-top: 1rem;
                margin-bottom: 1rem;
                border-radius: 0.375rem;
                transition: transform 0.2s ease;
            }
            
            .ai-explanation-box:hover {
                transform: translateY(-2px);
            }
            
            .ai-explanation-box p {
                color: var(--text-color);
                margin-bottom: 0;
            }

            /* Responsive adjustments */
            @media (max-width: 768px) {
                .dashboard-container {
                    padding: 0.75rem;
                }
                
                .card {
                    padding: 1rem;
                    margin-bottom: 1rem;
                }
                
                h1 { font-size: 1.5rem; }
                h2 { font-size: 1.25rem; }
                h3 { font-size: 1.125rem; }
                
                .summary-cards {
                    grid-template-columns: repeat(2, 1fr);
                }
                
                .insights-container {
                    grid-template-columns: 1fr;
                }
            }
            
            @keyframes loading-ring {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True) 