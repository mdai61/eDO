import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go



def plot_train_test_split(resampled_data):
    """
    Plot the train and test data with a clear indication of the train-test split,
    aligning the regions (A, B, C, D, E) with the y-axis.

    Parameters:
    resampled_data (pd.DataFrame): The processed data with 'TimeStamp' and 'Target' columns.
    """
    # Map the regions (A, B, C, D, E) to numeric values for plotting
    region_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    resampled_data['RegionNumeric'] = resampled_data['Target'].map(region_mapping)

    # Split the data into train and test sets
    train_data = resampled_data[resampled_data.index < '2024-11-01']
    test_data = resampled_data[resampled_data.index >= '2024-11-01']

    # Plot train and test data on the same graph
    plt.figure(figsize=(27, 12), dpi=300)  # High-resolution figure

    # Plot train data
    sns.lineplot(x=train_data.index, y=train_data['RegionNumeric'], label='Train Data', color='blue')

    # Plot test data
    sns.lineplot(x=test_data.index, y=test_data['RegionNumeric'], label='Test Data', color='orange')

    # Highlight the split point
    plt.axvline(x=pd.to_datetime('2024-11-01'), color='red', linestyle='--', linewidth=1.5, label='Train-Test Split')

    # Customize y-axis to show region names instead of numeric values
    plt.yticks(ticks=list(region_mapping.values()), labels=list(region_mapping.keys()), fontsize=14)

    # Add labels, legend, and title
    plt.title('Target with Train-Test Split (Aligned Regions)', fontsize=16)
    plt.xlabel('TimeStamp', fontsize=18)
    plt.ylabel('Region (Target)', fontsize=18)
    plt.xticks(rotation=45, fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('train_test_split_regions_plot.png', dpi=300)
    plt.show()



def plot_correlation_heatmap(resampled_data):
    """
    Plot a heatmap of the correlation matrix for selected features.

    Parameters:
    resampled_data (pd.DataFrame): The processed data.
    """
    # Calculate the correlation matrix
    correlation_matrix = resampled_data.drop(columns=['Target']).corr()

    # Create the heatmap using seaborn
    plt.figure(figsize=(15, 12))  # Adjust figure size if needed
    sns.heatmap(correlation_matrix, annot=True, cmap='YlOrBr', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()




def plot_train_test_split_dynamic(resampled_data):
    """
    Create an interactive plot of the train and test data with a clear indication 
    of the train-test split, aligning the regions (A, B, C, D, E) with the y-axis.

    Parameters:
    resampled_data (pd.DataFrame): The processed data with 'TimeStamp' and 'Target' columns.
    """
    # Map the regions (A, B, C, D, E) to numeric values for plotting
    region_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    resampled_data['RegionNumeric'] = resampled_data['Target'].map(region_mapping)

    # Split the data into train and test sets
    train_data = resampled_data[resampled_data.index < '2024-11-01']
    test_data = resampled_data[resampled_data.index >= '2024-11-01']

    # Create the figure
    fig = go.Figure()

    # Add train data trace
    fig.add_trace(
        go.Scatter(
            x=train_data.index,
            y=train_data['RegionNumeric'],
            mode='lines',
            name='Train Data',
            line=dict(color='blue')
        )
    )

    # Add test data trace
    fig.add_trace(
        go.Scatter(
            x=test_data.index,
            y=test_data['RegionNumeric'],
            mode='lines',
            name='Test Data',
            line=dict(color='orange')
        )
    )

    # Add train-test split line
    fig.add_trace(
        go.Scatter(
            x=['2024-11-01', '2024-11-01'],
            y=[1, 5],
            mode='lines',
            name='Train-Test Split',
            line=dict(color='red', dash='dash')
        )
    )

    # Customize the layout
    fig.update_layout(
        title='Target with Train-Test Split (Aligned Regions)',
        xaxis_title='TimeStamp',
        yaxis_title='Region (Target)',
        yaxis=dict(
            tickmode='array',
            tickvals=list(region_mapping.values()),
            ticktext=list(region_mapping.keys())
        ),
        legend=dict(font=dict(size=12)),
        template='plotly_white',
        width=1200,
        height=600
    )

    # Show the figure
    fig.show()


# Example usage:
# data_path = '/content/drive/My Drive/eDO_data_M1.csv'
# resampled_data = process_data(data_path)
# plot_train_test_split(resampled_data)
# plot_correlation_heatmap(resampled_data)
