from src.data_modules.base_dm import BaseDSBDM
from src.dataset import FilteredMNIST

class OneAndSevenSchrodingerDM(BaseDSBDM):
    def __init__(
        self,
        **kwargs,
    ):
        start_dataset = FilteredMNIST(
            download = True, 
            digit = 1,
            )
        
        end_dataset = FilteredMNIST(
            download = True,
            digit = 7,
            )
        
        super().__init__(
            start_dataset = start_dataset, 
            end_dataset = end_dataset, 
            **kwargs
            )