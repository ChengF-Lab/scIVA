# Dimensionality reduction and visualization of single-cell RNA-seq data with an improved deep variational autoencoder

## requirement
 numpy>=1.14.2
 pandas>=0.22.0
 scipy>=0.19.1
 scikit-learn>=0.19.1
 torch>=1.0.0
 tqdm>=4.28.1
 matplotlib>=3.0.2
 seaborn>=0.9.0

## Installation  

scIVA neural network is implemented in [Pytorch](https://pytorch.org/) framework.     
	

#### Input
* either a **count matrix file**:  
	* row is peak and column is barcode, in **txt**format

#### Output
Output will be saved in the output folder including:
* **feature.txt**:  latent feature representations of each cell used for clustering or visualization
* **embryo.eps**:  visualization of each cell
* ** The effect of clustering**  normalized mutual information (NMI), adjusted rand index (ARI) , completeness (COM), and homogeneity (HOM)      
#### Useful options  
* save results in a specific folder: [-o] or [--outdir] 
* modify the initial learning rate, default is 0.00001: [--lr]  
* change iterations by watching the convergence of loss, default is 20000: [-i] or [--max_iter]  
* run with scRNA-seq dataset: [--log_transform]
	
#### Help
Look for more usage of SCALE

	scIVA.py --help 

Use functions in SCALE packages.

	import sciva
	from sciva import *
	from sciva.plot import *
	from sciva.utils import *
	

#### Data availability  
Datasets were obtained from the Hemberg group (https://hemberg-lab.github.io/scRNA.seq.datasets/)

