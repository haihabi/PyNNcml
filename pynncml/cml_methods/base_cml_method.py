from torch import nn

from pynncml.cml_methods.results_data_structure import CMLResultsDataStructure
from pynncml.datasets.alignment import AttenuationType


class BaseCMLProcessingMethod(nn.Module):
    def __init__(self,input_data_type:AttenuationType,input_rate:int,output_rate:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data_type = input_data_type
        self.input_rate = input_rate
        self.output_rate = output_rate

    def convert_output_results(self,output_tensor)->CMLResultsDataStructure:
        raise NotImplemented
