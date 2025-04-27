import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import os
import logging
from PIL import Image
from wordcloud import WordCloud

modern_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='#1e293b',
        plot_bgcolor='#1e293b',
        font={'color': '#f1f5f9', 'family': 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif', 'size': 14},
        title={'font': {'color': '#f8fafc', 'size': 20}},
        xaxis={
            'gridcolor': '#334155', 
            'zerolinecolor': '#334155', 
            'tickfont': {'color': '#94a3b8', 'size': 12}, 
            'titlefont': {'color': '#f8fafc', 'size': 14}
        },
        yaxis={
            'gridcolor': '#334155', 
            'zerolinecolor': '#334155', 
            'tickfont': {'color': '#94a3b8', 'size': 12}, 
            'titlefont': {'color': '#f8fafc', 'size': 14}
        },
        legend={'font': {'color': '#f8fafc', 'size': 13}, 'bgcolor': 'rgba(30, 41, 59, 0.5)', 'bordercolor': '#334155'},
        colorway=['#10b981', '#3b82f6', '#ef4444', '#4f46e5', '#f59e0b', '#ec4899', '#8b5cf6'],
        hoverlabel={'font': {'family': 'Inter, sans-serif', 'size': 13}, 'bordercolor': 'rgba(0,0,0,0)'},
        margin={'t': 60, 'b': 50, 'l': 50, 'r': 30},
    )
)

#might edit
colors = {
    'positive': '#10b981', 
    'neutral': '#3b82f6',   # Blue
    'negative': '#ef4444',  # Red
    'primary': '#4f46e5',   
    'background': '#1e293b',
    'text': '#f8fafc',
    'subtext': '#94a3b8',
    'grid': '#334155'
}

def create_sentiment_breakdown(df, group_by='candidate'):
    """
    Create an enhanced stacked bar chart showing sentiment breakdown by group.
    
    Parameters:
    - df: DataFrame with sentiment analysis results
    - group_by: Column to group by (e.g., 'candidate', 'source')
    
    Returns:
    - Plotly figure
    """
    # CALCULATION
    grouped = df.groupby([group_by, 'sentiment']).size().reset_index(name='count')
    total_counts = df.groupby(group_by).size().reset_index(name='total')
    
    merged = pd.merge(grouped, total_counts, on=group_by)
    merged['percentage'] = (merged['count'] / merged['total']) * 100
    
    pivot_df = merged.pivot(index=group_by, columns='sentiment', values='percentage').reset_index()
    
    # NaN values
    for col in ['positive', 'neutral', 'negative']:
        if col not in pivot_df.columns:
            pivot_df[col] = 0
        else:
            pivot_df[col] = pivot_df[col].fillna(0)
    
    pivot_df = pd.merge(pivot_df, total_counts, on=group_by)
    
    fig = go.Figure()
    
    hovertemplate = '<b>%{x}</b><br>Positive: %{customdata[0]:.1f}% (%{customdata[1]} items)<br>Sample size: %{customdata[2]}<extra></extra>'
    
    pivot_df['positive_count'] = (pivot_df['positive'] * pivot_df['total'] / 100).round().astype(int)
    pivot_df['neutral_count'] = (pivot_df['neutral'] * pivot_df['total'] / 100).round().astype(int)
    pivot_df['negative_count'] = (pivot_df['negative'] * pivot_df['total'] / 100).round().astype(int)
    
    fig.add_trace(go.Bar(
        x=pivot_df[group_by],
        y=pivot_df['positive'],
        name='Positive',
        marker_color=colors['positive'],
        text=pivot_df['positive'].round(1).astype(str) + '%',
        textposition='inside',
        insidetextfont=dict(color='white', size=12, family='Inter, sans-serif'),
        customdata=np.column_stack((pivot_df['positive'], pivot_df['positive_count'], pivot_df['total'])),
        hovertemplate='<b>%{x}</b><br>Positive: %{y:.1f}% (%{customdata[1]} items)<br>Sample size: %{customdata[2]}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=pivot_df[group_by],
        y=pivot_df['neutral'],
        name='Neutral',
        marker_color=colors['neutral'],
        text=pivot_df['neutral'].round(1).astype(str) + '%',
        textposition='inside',
        insidetextfont=dict(color='white', size=12, family='Inter, sans-serif'),
        customdata=np.column_stack((pivot_df['neutral'], pivot_df['neutral_count'], pivot_df['total'])),
        hovertemplate='<b>%{x}</b><br>Neutral: %{y:.1f}% (%{customdata[1]} items)<br>Sample size: %{customdata[2]}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=pivot_df[group_by],
        y=pivot_df['negative'],
        name='Negative',
        marker_color=colors['negative'],
        text=pivot_df['negative'].round(1).astype(str) + '%',
        textposition='inside',
        insidetextfont=dict(color='white', size=12, family='Inter, sans-serif'),
        customdata=np.column_stack((pivot_df['negative'], pivot_df['negative_count'], pivot_df['total'])),
        hovertemplate='<b>%{x}</b><br>Negative: %{y:.1f}% (%{customdata[1]} items)<br>Sample size: %{customdata[2]}<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='stack',
        title={
            'text': f'Sentiment Breakdown by {group_by.title()}',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=None,
        yaxis_title='Percentage',
        legend_title='Sentiment',
        template=modern_template,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        hovermode='closest',
        uniformtext=dict(mode='hide', minsize=10)
    )
    




    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Reset View",
                        method="relayout",
                        args=[{"xaxis.range": [None, None], "yaxis.range": [0, 100]}]
                    )
                ],
                x=0.05,
                y=1.15,
            )
        ]
    )
    
    return fig

def create_sentiment_time_series(df, group_by='candidate', time_interval='D'):
    """
    Create an enhanced time series plot of sentiment over time.
    
    Parameters:
    - df: DataFrame with sentiment analysis results
    - group_by: Column to group by (default: 'candidate')
    - time_interval: Time interval for resampling ('D'=daily, 'W'=weekly)
    
    Returns:
    - Plotly figure
    """
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    
    if group_by is None:
        df_grouped = df.set_index('date')
        
        pos_counts = df_grouped[df_grouped['sentiment'] == 'positive'].resample(time_interval).size()
        neu_counts = df_grouped[df_grouped['sentiment'] == 'neutral'].resample(time_interval).size()
        neg_counts = df_grouped[df_grouped['sentiment'] == 'negative'].resample(time_interval).size()
        
        pos_counts = pos_counts.fillna(0)
        neu_counts = neu_counts.fillna(0)
        neg_counts = neg_counts.fillna(0)
        
        window = 3 if len(pos_counts) > 10 else 1
        
        fig.add_trace(
            go.Scatter(
                x=pos_counts.index, 
                y=pos_counts.rolling(window).mean(),
                mode='lines+markers',
                name='Positive',
                line=dict(color=colors['positive'], width=3, shape='spline'),
                marker=dict(color=colors['positive'], size=6, symbol='circle'),
                hoverlabel=dict(bgcolor=colors['positive']),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Count: %{y:.0f}<extra>Positive</extra>'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=neu_counts.index, 
                y=neu_counts.rolling(window).mean(),
                mode='lines+markers',
                name='Neutral',
                line=dict(color=colors['neutral'], width=3, shape='spline'),
                marker=dict(color=colors['neutral'], size=6, symbol='circle'),
                hoverlabel=dict(bgcolor=colors['neutral']),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Count: %{y:.0f}<extra>Neutral</extra>'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=neg_counts.index, 
                y=neg_counts.rolling(window).mean(),
                mode='lines+markers',
                name='Negative',
                line=dict(color=colors['negative'], width=3, shape='spline'),
                marker=dict(color=colors['negative'], size=6, symbol='circle'),
                hoverlabel=dict(bgcolor=colors['negative']),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Count: %{y:.0f}<extra>Negative</extra>'
            )
        )
        
        title = 'Sentiment Trends Over Time'
        
    else:
        groups = df[group_by].unique()
        
        candidate_colors = px.colors.qualitative.Vivid
        
        for i, group in enumerate(groups):
            group_df = df[df[group_by] == group].set_index('date')
            
            avg_polarity = group_df.resample(time_interval)['polarity'].mean()
            avg_polarity = avg_polarity.ffill()  
            
            counts = group_df.resample(time_interval).size()
            
            max_count = max(counts) if len(counts) > 0 else 1
            normalized_sizes = (counts / max_count * 20) + 5  
            
            color = candidate_colors[i % len(candidate_colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=avg_polarity.index,
                    y=avg_polarity.values,
                    mode='lines+markers',
                    name=group,
                    line=dict(color=color, width=3, shape='spline'),
                    marker=dict(
                        color=color, 
                        size=normalized_sizes, 
                        opacity=0.7,
                        line=dict(color='white', width=1)
                    ),
                    customdata=np.column_stack((counts, avg_polarity.values)),
                    hoverlabel=dict(bgcolor=color),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' + 
                                 f'{group}<br>' +
                                 'Sample size: %{customdata[0]:.0f}<br>' +
                                 'Avg. Polarity: %{customdata[1]:.2f}<extra></extra>'
                )
            )
        
        title = f'Sentiment Trends by {group_by.title()}'
    
    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Date',
        yaxis_title='Sentiment Value' if group_by else 'Count',
        template=modern_template,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        hovermode='closest',
    )
    
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider_thickness=0.05,
        rangeslider=dict(bgcolor='#334155'),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all")
            ]),
            bgcolor='#334155',
            activecolor='#6366f1'
        )
    )
    
    return fig

def create_polarity_distribution(df, group_by='candidate'):
    """
    Create an enhanced violin plot showing the distribution of sentiment polarity scores.
    
    Parameters:
    - df: DataFrame with sentiment analysis results
    - group_by: Column to group by (default: 'candidate')
    
    Returns:
    - Plotly figure
    """
    fig = go.Figure()
    
    groups = df[group_by].unique()
    
    for i, group in enumerate(groups):
        group_data = df[df[group_by] == group]
        
        color = px.colors.qualitative.Vivid[i % len(px.colors.qualitative.Vivid)]
        
        fig.add_trace(
            go.Violin(
                x=[group] * len(group_data),
                y=group_data['polarity'],
                name=group,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                opacity=0.6,
                line_color=color,
                marker=dict(
                    color=color,
                    opacity=0.3,
                    size=8,
                    symbol='circle'
                ),
                points='all',
                jitter=0.5,
                customdata=group_data[['sentiment', 'feedback']],
                hovertemplate='<b>%{x}</b><br>' +
                               'Polarity: %{y:.3f}<br>' +
                               'Sentiment: %{customdata[0]}<br>' +
                               'Feedback: %{customdata[1]}<extra></extra>'
            )
        )
    
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(groups) - 0.5,
        y1=0,
        line=dict(color="#f8fafc", width=1, dash="dash")
    )
    
    fig.add_annotation(
        x=len(groups) - 0.5,
        y=0,
        xref="x",
        yref="y",
        text="Neutral",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#f8fafc",
        arrowsize=1,
        arrowwidth=1,
        ax=30,
        ay=0,
        font=dict(color="#f8fafc", size=10)
    )
    




    fig.update_layout(
        title={
            'text': f'Distribution of Sentiment Polarity by {group_by.title()}',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=group_by.title(),
        yaxis_title='Polarity Score',
        template=modern_template,
        margin=dict(l=40, r=40, t=60, b=40),
        height=450,
        hovermode='closest',
        xaxis=dict(
            tickangle=-20
        )
    )
    
    fig.add_annotation(
        x=len(groups) / 2,
        y=0.95,
        xref="paper",
        yref="paper",
        text="Polarity Score: -1 (Very Negative) to +1 (Very Positive)",
        showarrow=False,
        font=dict(color="#94a3b8", size=12)
    )
    
    return fig

def create_sentiment_pie_chart(dataframe):
    """
    Create a pie chart showing the distribution of sentiment
    """
    if dataframe is None or dataframe.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for sentiment breakdown",
            font={"color": colors['subtext'], "size": 16},
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(template=modern_template)
        return fig
    
    sentiment_counts = dataframe['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    
    total = sentiment_counts['count'].sum()
    sentiment_counts['percentage'] = (sentiment_counts['count'] / total * 100).round(1)
    
    color_mapping = {
        'positive': colors['positive'],
        'neutral': colors['neutral'],
        'negative': colors['negative']
    }
    
    hovertemplate = '<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata}%<extra></extra>'
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts['sentiment'],
        values=sentiment_counts['count'],
        customdata=sentiment_counts['percentage'],
        hovertemplate=hovertemplate,
        marker=dict(
            colors=[color_mapping.get(sentiment, colors['primary']) for sentiment in sentiment_counts['sentiment']],
            line=dict(color=colors['background'], width=1.5)
        ),
        textinfo='label+percent',
        textfont=dict(size=14, color=colors['text']),
        insidetextorientation='radial',
        pull=[0.03, 0.03, 0.03],  
        hole=0.4  
    )])
    
    fig.update_layout(
        template=modern_template,
        title="Sentiment Distribution",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        annotations=[dict(
            text="Sentiment<br>Breakdown",
            x=0.5, y=0.5,
            font=dict(size=15, color=colors['text']),
            showarrow=False
        )]
    )
    
    return fig

def create_time_series(dataframe, candidate=None):
    """
    Create a time series visualization of sentiment over time
    """
    if dataframe is None or dataframe.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for time series visualization",
            font={"color": colors['subtext'], "size": 16},
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(template=modern_template)
        return fig
    
    df = dataframe.copy()
    
    if candidate and candidate in df['candidate'].unique():
        df = df[df['candidate'] == candidate]
    
    if 'timestamp' not in df.columns or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No timestamp data available for time series visualization",
            font={"color": colors['subtext'], "size": 16},
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(template=modern_template)
        return fig
    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df_by_day = df.set_index('timestamp')
    
    sentiment_by_day = df_by_day.groupby([pd.Grouper(freq='D'), 'sentiment']).size().unstack(fill_value=0)
    
    date_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D')
    sentiment_by_day = sentiment_by_day.reindex(date_range, fill_value=0)
    
    sentiment_by_day = sentiment_by_day.ffill()
    
    fig = go.Figure()
    
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in sentiment_by_day.columns:
            fig.add_trace(go.Scatter(
                x=sentiment_by_day.index,
                y=sentiment_by_day[sentiment],
                mode='lines+markers',
                name=sentiment.capitalize(),
                line=dict(width=3, color=colors[sentiment], shape='spline', smoothing=1.3),
                marker=dict(
                    size=8,
                    symbol='circle',
                    opacity=0.8,
                    line=dict(width=1, color=colors['background'])
                ),
                hovertemplate='<b>%{x}</b><br>%{y} ' + sentiment + ' responses<extra></extra>'
            ))
    
    if not sentiment_by_day.empty:
        total_by_day = sentiment_by_day.sum(axis=1)
        if len(total_by_day) >= 7: 
            total_ma = total_by_day.rolling(window=7).mean()
            
            fig.add_trace(go.Scatter(
                x=total_ma.index,
                y=total_ma,
                mode='lines',
                name='7-Day Average (Total)',
                line=dict(width=2, color='rgba(255, 255, 255, 0.7)', dash='dash'),
                hovertemplate='<b>%{x}</b><br>7-Day Avg: %{y:.1f} responses<extra></extra>'
            ))
    
    title = f"Sentiment Trends Over Time{' for ' + candidate if candidate else ''}"
    
    fig.update_layout(
        template=modern_template,
        title=title,
        xaxis_title="Date",
        yaxis_title="Number of Responses",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_keyword_chart(keywords_dict, display_count=10):
    """
    Create a horizontal bar chart of keywords and their frequencies
    
    Parameters:
    - keywords_dict: Dictionary with keywords and their frequencies, or list of (keyword, frequency) tuples
    - display_count: Number of keywords to display
    
    Returns:
    - Plotly figure containing the bar chart
    """
    if keywords_dict is None or not keywords_dict:
        fig = go.Figure()
        fig.add_annotation(
            text="No keyword data available",
            font={"color": colors['subtext'], "size": 16},
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(template=modern_template)
        return fig
    
    if isinstance(keywords_dict, dict):
        keywords = sorted(keywords_dict.items(), key=lambda x: x[1], reverse=True)[:display_count]
    elif isinstance(keywords_dict, list):
        keywords = sorted(keywords_dict, key=lambda x: x[1], reverse=True)[:display_count]
    else:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Invalid input type: {type(keywords_dict).__name__}. Expected dict or list.",
            font={"color": colors['subtext'], "size": 16},
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(template=modern_template)
        return fig
    
    words, counts = zip(*keywords)
    
    max_count = max(counts) if counts else 1
    
    if max_count > 0:
        color_scale = [(count / max_count) * 0.8 for count in counts]  
    else:
        color_scale = [0.5] * len(counts) 
    
    colors_arr = [f'rgba(99, 102, 241, {0.4 + scale})' for scale in color_scale]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=words,
        x=counts,
        orientation='h',
        marker=dict(
            color=colors_arr,
            line=dict(color=colors['primary'], width=1.5)
        ),
        hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>',
        texttemplate='%{x}',
        textposition='outside',
        textfont=dict(color=colors['text'])
    ))
    
    fig.update_layout(
        template=modern_template,
        title="Top Keywords",
        xaxis_title="Frequency",
        yaxis=dict(
            title=None,
            autorange="reversed"  
        ),
        margin=dict(l=10, r=10, t=60, b=50)
    )
    
    return fig

def create_source_comparison(df):
    """
    Create an enhanced comparison of feedback sources.
    
    Parameters:
    - df: DataFrame with sentiment analysis results
    
    Returns:
    - Plotly figure
    """
    try:
        if df is None or df.empty:

            fig = go.Figure()
            fig.add_annotation(
                text="No data available for visualization",
                showarrow=False,
                font=dict(size=16),
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )
            return fig
            
      
        if 'source' not in df.columns or 'sentiment' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required columns: 'source' or 'sentiment'",
                showarrow=False,
                font=dict(size=16),
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )
            return fig
            
        source_sentiment = df.groupby(['source', 'sentiment']).size().reset_index(name='count')
        
        if source_sentiment.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No source or sentiment data to visualize",
                showarrow=False,
                font=dict(size=16),
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )
            return fig
        
        source_totals = df.groupby('source').size().reset_index(name='total')
        source_sentiment = pd.merge(source_sentiment, source_totals, on='source')
        source_sentiment['percentage'] = source_sentiment['count'] / source_sentiment['total'] * 100
        
        unique_sources = source_sentiment['source'].unique()
        labels = unique_sources.tolist() + source_sentiment['sentiment'].tolist()
        
        parents = [''] * len(unique_sources)
        for source in source_sentiment['source']:
            parents.append(source)
            
        values = [0] * len(unique_sources) + source_sentiment['count'].tolist()
        
        source_colors = ['rgba(99, 102, 241, 0.8)'] * len(unique_sources)
        sentiment_colors = []
        for sentiment in source_sentiment['sentiment']:
            if sentiment == 'positive':
                sentiment_colors.append(colors['positive'])
            elif sentiment == 'neutral':
                sentiment_colors.append(colors['neutral'])
            else:
                sentiment_colors.append(colors['negative'])
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            marker=dict(
                colors=source_colors + sentiment_colors,
                line=dict(width=1, color="#334155")
            ),
            textinfo='label+percent parent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.1f}%<extra></extra>',
            hoverlabel=dict(bgcolor='#293548', font_size=12, font_family="Inter, sans-serif")
        ))
        
        fig.update_layout(
            title={
                'text': 'Sentiment Distribution by Source',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            template=modern_template,
            margin=dict(l=10, r=10, t=60, b=10),
            height=500
        )
        
        return fig
    except Exception as e:
        print(f"Error creating source comparison: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            showarrow=False,
            font=dict(size=14),
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(template=modern_template)
        return fig

def create_subjectivity_vs_polarity(df, group_by='candidate'):
    """
    Create an enhanced scatter plot of subjectivity vs. polarity.
    
    Parameters:
    - df: DataFrame with sentiment analysis results
    - group_by: Column to group by (default: 'candidate')
    
    Returns:
    - Plotly figure
    """
    fig = px.scatter(
        df,
        x='subjectivity',
        y='polarity',
        color=group_by,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        size='subjectivity',
        size_max=15,
        opacity=0.7,
        hover_data=['feedback', 'sentiment'],
        template=modern_template
    )
    

    fig.add_shape(
        type="rect",
        x0=0,
        y0=0.05,
        x1=1,
        y1=1,
        line=dict(width=0),
        fillcolor="rgba(54, 211, 153, 0.1)",
        layer="below"
    )
    
    fig.add_shape(
        type="rect",
        x0=0,
        y0=-1,
        x1=1,
        y1=-0.05,
        line=dict(width=0),
        fillcolor="rgba(255, 107, 107, 0.1)",
        layer="below"
    )
    
    fig.add_shape(
        type="rect",
        x0=0,
        y0=-0.05,
        x1=1,
        y1=0.05,
        line=dict(width=0),
        fillcolor="rgba(77, 171, 247, 0.1)",
        layer="below"
    )
    
    fig.add_annotation(
        x=0.95,
        y=0.8,
        text="Positive",
        showarrow=False,
        font=dict(color="rgba(54, 211, 153, 0.7)", size=14)
    )
    
    fig.add_annotation(
        x=0.95,
        y=-0.8,
        text="Negative",
        showarrow=False,
        font=dict(color="rgba(255, 107, 107, 0.7)", size=14)
    )
    
    fig.add_annotation(
        x=0.95,
        y=0,
        text="Neutral",
        showarrow=False,
        font=dict(color="rgba(77, 171, 247, 0.7)", size=14)
    )
    
    fig.update_layout(
        title={
            'text': f'Sentiment Polarity vs. Subjectivity by {group_by.title()}',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Subjectivity (0=Objective, 1=Subjective)',
        yaxis_title='Polarity (-1=Negative, 1=Positive)',
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
        hovermode='closest',
        legend=dict(
            title=group_by.title(),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=0,
        line=dict(color="#94a3b8", width=1, dash="dash")
    )
    
    return fig 

def create_wordcloud(text_data, mask_path=None, background_color='#1e293b', 
                     colormap='viridis', title="Word Cloud Visualization"):
    """
    Create a word cloud visualization from text data with enhanced styling.
    
    Parameters:
    - text_data: String or list of strings containing the text to visualize
    - mask_path: Path to an image file to use as mask (optional)
    - background_color: Background color for the word cloud
    - colormap: Matplotlib colormap name to use for word colors
    - title: Title for the word cloud visualization
    
    Returns:
    - Plotly figure containing the wordcloud image
    """
    if not text_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No text data available for word cloud visualization",
            font={"color": colors['subtext'], "size": 16},
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(template=modern_template)
        return fig
    
    if isinstance(text_data, list):
        text = ' '.join(text_data)
    else:
        text = text_data
        
    mask = None
    if mask_path and os.path.exists(mask_path):
        try:
            mask = np.array(Image.open(mask_path))
        except Exception as e:
            logging.warning(f"Failed to load mask image: {e}")

    try:
        wc = WordCloud(
            background_color=background_color,
            max_words=150,
            mask=mask,
            colormap=colormap,
            width=800,
            height=500,
            prefer_horizontal=0.9,
            min_font_size=8,
            max_font_size=80,
            random_state=42,
            contour_width=1,
            contour_color='rgba(255, 255, 255, 0.2)',
            relative_scaling=0.6,  
            collocations=True, 
            regexp=r'\w[\w\' ]+' 
        )
        
        wc.generate_from_text(text)
        
        img = wc.to_image()
        
        img_array = np.array(img)
        
        fig = px.imshow(img_array)
        
        fig.update_layout(
            template=modern_template,
            title=title,
            margin=dict(l=10, r=10, t=60, b=10),
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            coloraxis_showscale=False
        )
        
        fig.update_traces(hoverinfo='none', hovertemplate=None)
        
        return fig
    
    except Exception as e:
        logging.error(f"Error generating word cloud: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error generating word cloud: {str(e)}",
            font={"color": colors['subtext'], "size": 14},
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(template=modern_template)
        return fig 

def create_choropleth_map(df, location_col, value_col, scope="usa", title="Geographic Distribution", 
                          color_scale="Viridis", location_mode="USA-states"):
    """
    Create an enhanced choropleth map visualization for geographic data distribution.
    
    Parameters:
    - df: DataFrame containing the geographic data
    - location_col: Column name containing location identifiers (state abbrev, FIPS, etc.)
    - value_col: Column name containing values to be visualized
    - scope: Geographic scope for the map (default: "usa")
    - title: Title for the visualization
    - color_scale: Color scale for the map (default: "Viridis")
    - location_mode: Mode for location identifiers (default: "USA-states")
    
    Returns:
    - Plotly figure containing the choropleth map
    """
    if df.empty or not all(col in df.columns for col in [location_col, value_col]):
        fig = go.Figure()
        fig.add_annotation(
            text="No geographic data available for visualization",
            font={"color": colors['subtext'], "size": 16},
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(template=modern_template)
        return fig
    
    geo_data = df.copy()
    
    if geo_data[location_col].dtype == 'object':
        geo_data[location_col] = geo_data[location_col].str.upper()
    
    agg_data = geo_data.groupby(location_col)[value_col].sum().reset_index()
    
    vmin = agg_data[value_col].min()
    vmax = agg_data[value_col].max()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Choropleth(
            locations=agg_data[location_col],
            z=agg_data[value_col],
            locationmode=location_mode,
            colorscale=color_scale,
            colorbar=dict(
                title=value_col,
                thickness=20,
                len=0.7,
                bgcolor='rgba(255,255,255,0.1)',
                borderwidth=0,
                outlinewidth=0,
                tickfont=dict(color=colors['text']),
                titlefont=dict(color=colors['text'])
            ),
            marker=dict(
                line=dict(
                    color='rgba(255,255,255,0.3)',
                    width=0.5
                )
            ),
            zmin=vmin,
            zmax=vmax,
            hovertemplate='<b>%{location}</b><br>%{z:,.0f} ' + value_col + '<extra></extra>'
        )
    )
    
    fig.update_layout(
        template=modern_template,
        title=dict(
            text=title,
            font=dict(color=colors['title'], size=22)
        ),
        geo=dict(
            scope=scope,
            projection=dict(type='albers usa' if scope == 'usa' else 'natural earth'),
            showlakes=True,
            lakecolor='rgba(0,87,156,0.3)',
            showland=True,
            landcolor='rgba(240,240,240,0.2)',
            subunitcolor='rgba(255,255,255,0.2)',
            countrycolor='rgba(255,255,255,0.2)',
            coastlinecolor='rgba(255,255,255,0.2)',
            showcoastlines=True,
            showframe=False,
            framecolor='rgba(255,255,255,0.2)',
            showocean=True,
            oceancolor='rgba(20,20,60,0.2)'
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0, t=60, b=0),
        autosize=True,
        height=550
    )
    
    return fig 

def create_topic_bubble_chart(topics_data, topic_labels=None, title="Topic Distribution", colorscale="Viridis"):
    """
    Create an interactive bubble chart for visualizing topic modeling results.
    
    Parameters:
    - topics_data: DataFrame or dict containing topic data with 'x', 'y', 'size', and 'topic_id' columns
                  If dict, should have format {topic_id: {'x': x_val, 'y': y_val, 'size': size_val, 'top_terms': [terms]}}
    - topic_labels: Optional dict mapping topic_ids to human-readable labels
    - title: Title for the visualization
    - colorscale: Color scale for the bubbles
    
    Returns:
    - Plotly figure containing the bubble chart
    """
    if topics_data is None or (isinstance(topics_data, pd.DataFrame) and topics_data.empty) or (isinstance(topics_data, dict) and not topics_data):
        fig = go.Figure()
        fig.add_annotation(
            text="No topic data available for visualization",
            font={"color": colors['subtext'], "size": 16},
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(template=modern_template)
        return fig
    
    # Convert  to DataFrame if needed????
    if isinstance(topics_data, dict):
        topics_list = []
        for topic_id, data in topics_data.items():
            topic_entry = {
                'topic_id': topic_id,
                'x': data.get('x', 0),
                'y': data.get('y', 0),
                'size': data.get('size', 1),
                'top_terms': data.get('top_terms', [])
            }
            topics_list.append(topic_entry)
        topics_df = pd.DataFrame(topics_list)
    else:
        topics_df = topics_data.copy()
        
    required_cols = ['topic_id', 'x', 'y', 'size']
    if not all(col in topics_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in topics_df.columns]
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required columns: {', '.join(missing)}",
            font={"color": colors['subtext'], "size": 16},
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(template=modern_template)
        return fig
        
    if topic_labels and isinstance(topic_labels, dict):
        topics_df['label'] = topics_df['topic_id'].map(lambda x: topic_labels.get(x, f"Topic {x}"))
    else:
        topics_df['label'] = topics_df['topic_id'].apply(lambda x: f"Topic {x}")
    
    max_size = topics_df['size'].max()
    min_size = topics_df['size'].min()
    size_range = max_size - min_size
    
    if size_range > 0:
        topics_df['bubble_size'] = ((topics_df['size'] - min_size) / size_range * 50) + 10
    else:
        topics_df['bubble_size'] = 30 
    
    if 'top_terms' in topics_df.columns:
        topics_df['hover_text'] = topics_df.apply(
            lambda row: f"{row['label']}<br>Size: {row['size']}<br><br>Top terms: {', '.join(row['top_terms'][:5])}" 
            if isinstance(row.get('top_terms'), list) and row['top_terms'] 
            else f"{row['label']}<br>Size: {row['size']}", 
            axis=1
        )
    else:
        topics_df['hover_text'] = topics_df.apply(lambda row: f"{row['label']}<br>Size: {row['size']}", axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=topics_df['x'],
        y=topics_df['y'],
        mode='markers+text',
        marker=dict(
            size=topics_df['bubble_size'],
            color=topics_df.index,
            colorscale=colorscale,
            line=dict(width=1.5, color='rgba(255,255,255,0.3)'),
            opacity=0.8,
            colorbar=dict(
                title="Topic ID",
                thickness=15,
                len=0.7,
                bgcolor='rgba(255,255,255,0.1)',
                borderwidth=0,
                tickfont=dict(color=colors['text'])
            ),
        ),
        text=topics_df['label'],
        textposition="top center",
        textfont=dict(family="Inter, sans-serif", size=11, color=colors['text']),
        hoverinfo='text',
        hovertext=topics_df['hover_text'],
        hoverlabel=dict(
            bgcolor='rgba(50,50,50,0.9)',
            bordercolor='rgba(255,255,255,0.2)',
            font=dict(family="Inter, sans-serif", size=12, color='white')
        )
    ))
    
    fig.update_layout(
        template=modern_template,
        title=dict(
            text=title,
            font=dict(color=colors['title'], size=22)
        ),
        xaxis=dict(
            title="Dimension 1",
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)',
            showticklabels=False
        ),
        yaxis=dict(
            title="Dimension 2",
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)',
            showticklabels=False
        ),
        hovermode='closest',
        showlegend=False,
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=20, r=20, t=60, b=20),
        autosize=True,
        height=600
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text="• Bubble size represents topic prevalence<br>• Position shows relationship between topics",
        showarrow=False,
        font=dict(size=11, color=colors['subtext']),
        align="left",
        bgcolor="rgba(0,0,0,0.3)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        borderpad=4
    )
    
    return fig 