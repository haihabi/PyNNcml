
from torch import nn
import numpy as np
import xarray as xr

from pynncml.cml_methods.base_cml_method import BaseCMLProcessingMethod
from pynncml.cml_methods.results_data_structure import CMLResultsDataStructure
from pynncml.datasets.xarray_processing import xarray2link
from pynncml.multiple_cmls_methods import InferMultipleCMLs


class XarrayInferenceEngine(nn.Module):
    def __init__(self,in_cml2rain_method:BaseCMLProcessingMethod,is_recurrent=True,is_attenuation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_engine = InferMultipleCMLs(in_cml2rain_method,is_recurrent,is_attenuation)


    def forward(self, x_xarray):
        link_set=xarray2link(x_xarray)
        results_data= self.inference_engine(link_set)

        new_var_dims = ('time', 'sublink_id', 'cml_id')
        new_var_coords = {
            'time': x_xarray.time,
            'sublink_id': x_xarray.sublink_id,
            'cml_id': x_xarray.cml_id
        }
        for rname in CMLResultsDataStructure.results_types_list():
            x_xarray[rname] = xr.DataArray(
            np.full((x_xarray.sizes['time'], x_xarray.sizes['sublink_id'], x_xarray.sizes['cml_id']), np.nan),
            coords=new_var_coords,
            dims=new_var_dims
            )

        for i, link in enumerate(link_set):
            results=self.inference_engine.cml2rain.convert_output_results(results_data[i])
            for rname in CMLResultsDataStructure.results_types_list():
                # Create a new DataArray with the correct dimensions and coordinates
                new_values = xr.DataArray(
                    getattr(results,rname),
                    coords=[x_xarray.time],
                    dims=['time']
                )
                # Assign the new values to the dataset at the specified slice.
                # This will create a new variable if it doesn't exist.
                x_xarray[rname].loc[dict(sublink_id=link.sublink_id, cml_id=link.cml_id)] = new_values

        return x_xarray