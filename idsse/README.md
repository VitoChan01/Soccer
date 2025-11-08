# An integrated dataset of spatiotemporal and event data in elite soccer

This is the official repository for the paper: 
>Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of
> spatiotemporal and event data in elite soccer. *Scientific Data, 12*(1), 195. https://doi.org/10.1038/s41597-025-04505-y

## Project Structure

- `data_processing.py`: Functions for loading and processing metadata, event data, and position data.
- `visualization.py`: Functions for visualizing the processed data.
- `data_summary.ipynb`: Jupyter notebook to replicate the descriptive statistics and visualizations presented in the paper.

## Data Source and Characteristics

- Soccer matches from the [German Bundesliga](https://www.dfl.de/de/) (1st and 2nd divisions) 
- Size: 7 full matches
  - Official metadata (match information)
  - Official event data.
  - Official position data captured by [TRACAB](https://tracab.com/products/tracab-technologies/)
  
## License
The data are provided with authorization of the [Deutsche Fussball Liga (DFL)](https://www.dfl.de/de/). The dataset
is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). You must therefore give appropriate credit
when using this dataset by
1) naming the *Deutsche Fußball Liga (DFL)*
2) citing this [publication](https://doi.org/10.1038/s41597-025-04505-y)

## Usage

### Data Processing and visualization

1. Download the raw data [here](https://doi.org/10.6084/m9.figshare.28196177)
2. Open the `data_summary.ipynb` notebook.
3. Define the path to your dataset directory in the `path` variable.
4. Run the cells to load and process the data.
5. The processed data summary will be displayed.

## Citation
```BibTeX
@article{bassek2025integrated,
  title={An integrated dataset of spatiotemporal and event data in elite soccer},
  author={Bassek, Manuel and Rein, Robert and Weber, Hendrik and Memmert, Daniel},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={195},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
---

## Funding
This project has been kindly supported by the [Institute of Exercise Training and Sport
Informatics](https://www.dshs-koeln.de/en/institut-fuer-trainingswissenschaft-und-sportinformatik/) at the German Sport
University Cologne, under supervision of Prof. Daniel Memmert. Funding was provided by the 
[German Research Foundation (DFG)](https://www.dfg.de/en) 
([*floodlight*](https://gepris.dfg.de/gepris/projekt/522904388?context=projekt&task=showDetail&id=522904388&)).
