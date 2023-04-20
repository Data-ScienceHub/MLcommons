"""
Done following
https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/models/base_model.html#BaseModel.plot_prediction
"""

import os, sys
import numpy as np
from pandas import DataFrame, to_timedelta
from typing import List, Dict
import matplotlib.pyplot as plt

sys.path.append('..')
from script.utils import calculate_result
from Class.PredictionProcessor import *
from Class.PlotConfig import *

from matplotlib.ticker import MultipleLocator

class PlotResults:
    def __init__(self, figPath:str, targets:List[str], figsize=FIGSIZE, show=True) -> None:
        self.figPath = figPath
        if not os.path.exists(figPath):
            print(f'Creating folder {figPath}')
            os.makedirs(figPath, exist_ok=True)

        self.figsize = figsize
        self.show = show
        self.targets = targets
    
    def plot(
        self, df:DataFrame, target:str, title:str=None, scale=1, 
        base:int=None, figure_name:str=None, plot_error:bool=False
    ):
        fig, ax = plt.subplots(figsize=self.figsize)
        if title is not None: plt.title(title)
        x_column = 'Date'

        plt.plot(df[x_column], df[target], color='blue', label='Ground Truth')
        plt.plot(df[x_column], df[f'Predicted_{target}'], color='green', label='Prediction')

        if plot_error:
            plt.plot(df[x_column], abs(df[target] - df[f'Predicted_{target}']), color='red', label='Error')
        _, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max*1.1)
        
        if base is None:
            x_first_tick = df[x_column].min()
            x_last_tick = df[x_column].max()
            x_major_ticks = DATE_TICKS
            ax.set_xticks(
                [x_first_tick + (x_last_tick - x_first_tick) * i / (x_major_ticks - 1) for i in range(x_major_ticks)]
            )
        else:
            # plt.xlim(left=df[x_column].min() - ONE_DAY, right=df[x_column].max() + ONE_DAY)
            ax.xaxis.set_major_locator(MultipleLocator(base=base))
        
        # plt.xticks(rotation = 15)
        # plt.xlabel(x_column)

        if scale>1:
            if scale==1e3 or scale==1e6:
                label_text = [] 
                if scale ==1e3: unit = 'K'
                else: unit = 'M'

                for loc in ax.get_yticks():
                    if loc == 0:
                        label_text.append('0')
                    else:
                        label_text.append(f'{loc/scale:0.5g}{unit}') 

                ax.set_yticks(ax.get_yticks())
                ax.set_yticklabels(label_text)
                plt.ylabel(f'Daily {target}')
            else:
                ax.yaxis.set_major_formatter(get_formatter(scale))
                if scale==1e3: unit = 'in thousands'
                elif scale==1e6: unit = 'in millions'
                else: unit = f'x {scale:.0e}'

                plt.ylabel(f'Daily {target} ({unit})')
        else:
            plt.ylabel(f'Daily {target}')
            
        if plot_error:
            plt.legend(framealpha=0.3, edgecolor="black", ncol=3, loc='best')
        else:
            plt.legend(framealpha=0.3, edgecolor="black", ncol=2, loc='best')
            
        # fig.tight_layout() # might change the y axis unexpectedly

        if figure_name is not None:
            plt.savefig(os.path.join(self.figPath, figure_name), dpi=DPI)
        if self.show:
            plt.show()
        return fig

    def summed_plot(
        self, merged_df:DataFrame, type:str='', save:bool=True, 
        base:int=None, plot_error:bool=False
    ):
        """
        Plots summation of prediction and observation from all counties

        Args:
            figure_name: must contain the figure type extension. No need to add target name as 
            this method will add the target name as prefix to the figure name.
        """
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        summed_df = PredictionProcessor.makeSummed(merged_df, self.targets)
        figures = []
        for target in self.targets:
            predicted_column = f'Predicted_{target}'
            y_true, y_pred = merged_df[target].values, merged_df[predicted_column].values
            
            mae, rmse, rmsle, smape, nnse = calculate_result(y_true, y_pred)
            title = f'MAE {mae:0.3g}, RMSE {rmse:0.4g}, RMSLE {rmsle:0.3g}, SMAPE {smape:0.3g}, NNSE {nnse:0.3g}'
            
            if (summed_df[target].max() - summed_df[target].min()) >= 1e3:
                scale = 1e3
            else: scale = 1

            target_figure_name = None
            if save: target_figure_name = f'Summed_plot_{target}_{type}.jpg'

            fig = self.plot(
                summed_df, target, title, scale, base, target_figure_name, plot_error
            )
            figures.append(fig)
        
        return figures

    def individual_plot(
        self, df:DataFrame, fips:str, type:str='', save:bool=True, 
        base:int=None, plot_error:bool=False
    ):
        """
        Plots the prediction and observation for this specific county

        Args:
            figure_name: must contain the figure type extension. No need to add target name as 
            this method will add the target name as prefix to the figure name.
        """

        assert fips in df['FIPS'].values, f'Provided FIPS code {fips} does not exist in the dataframe.'
        df = df[df['FIPS']==fips]

        figures = []
        for target in self.targets:
            predicted_column = f'Predicted_{target}'
            y_true, y_pred = df[target].values, df[predicted_column].values
            
            mae, rmse, smape, nnse = calculate_result(y_true, y_pred)
            if (df[target].max() - df[target].min())>=2e3: scale = 1e3
            else: scale = 1

            target_figure_name = None
            if save: target_figure_name = f'Individual_plot_{target}_{type}_FIPS_{fips}.jpg'
            
            title = f'{target} MAE {mae:0.4g}, RMSE {rmse:0.4g}, SMAPE {smape:0.4g}, NNSE {nnse:0.4g}'
            fig = self.plot(df, target, title, scale, base, target_figure_name, plot_error)
            figures.append(fig)

        return figures