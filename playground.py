from src.dataset import FilteredByAttrCelebA

dataset = FilteredByAttrCelebA(attr=32, on_or_off=0)
print(dataset.unique_identifier)