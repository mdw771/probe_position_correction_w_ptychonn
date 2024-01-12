# How to prepare data for PtychoShelves

1. Use `pack_data.py` to extract diffraction data, positions, and probe functions into a folder `outputs/packed_data_for_ptychoshelves`. For each scan index, 3 folders will be created for true positions, baseline positions, and calculated
   positions. 
2. `cd outputs/packed_data_for_ptychoshelves`. 
3. Use `python converttomat.py test*` to convert data to `*.hdf5` and `*.mat` used by PtychoShelves. 
