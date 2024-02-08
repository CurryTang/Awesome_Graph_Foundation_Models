# Towards-Graph-Foundation-Models-New-perspective
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 

Accompanied repositories for our paper [Graph foundation model](https://arxiv.org/abs/2402.02216)


## Theoretical backgrounds for GFM development

A curated list of papers grounding the theoretical foundations of GFMs

### General Theory

* **Geometric ML Book** Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges [[Book](https://arxiv.org/abs/2104.13478)]
* (**ICML '23**) On Over-Squashing in Message Passing Neural Networks: The Impact of Width, Depth, and Topology [[Paper](https://arxiv.org/abs/2302.02941)]
* (**ICLR '22**) Understanding over-squashing and bottlenecks on graphs via curvature [[Paper](Understanding over-squashing and bottlenecks on graphs via curvature)]
* (**ICLR '21**) HOW NEURAL NETWORKS EXTRAPOLATE: FROM FEEDFORWARD TO GRAPH NEURAL NETWORKS [[Paper](https://arxiv.org/pdf/2009.11848.pdf)]
* (**ICMLW '20** A Note on Over-Smoothing for Graph Neural Networks [[Paper](https://arxiv.org/abs/2006.13318)]
* (**ICLR '20**) WHAT CAN NEURAL NETWORKS REASON ABOUT? [[Paper](https://arxiv.org/pdf/1905.13211.pdf)]
* (**ICLR '20**) The Logical Expressiveness of Graph Neural Networks  [[Paper](https://openreview.net/forum?id=r1lZ7AEKvB)]

### Node-level tasks (Node classification)

* (**Arxiv '24**) Understanding Heterophily for Graph Neural Networks [[Paper](https://arxiv.org/abs/2401.09125)]
* ![image](https://github.com/CurryTang/Towards-Graph-Foundation-Models-New-perspective-/assets/15672123/89a23a37-71d4-47f7-8949-7d859a41e369)
 (**NIPS '23**) Demystifying Structural Disparity in Graph Neural Networks: Can One Size Fit All? [[Paper](https://arxiv.org/abs/2306.01323)]
* (**NIPS '23**) When Do Graph Neural Networks Help with Node Classification? Investigating the Impact of Homophily Principle on Node Distinguishability [[Paper](https://arxiv.org/abs/2304.14274)]
* (**NIPS '23**) Demystifying Oversmoothing in Attention-Based Graph Neural Networks [[Paper](https://arxiv.org/pdf/2305.16102.pdf)]
* (**NIPS '22**) Revisiting Heterophily For Graph Neural Networks [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/092359ce5cf60a80e882378944bf1be4-Abstract-Conference.html)]
* (**ICLR '22**) Is Homophily a Necessity for Graph Neural Networks? [[Paper](https://arxiv.org/abs/2106.06134)]
* (**ICLR '20**) Graph Neural Networks Exponentially Lose Expressive Power for Node Classification [[Paper](https://arxiv.org/pdf/1905.10947)]
* (**Annual Review of Sociology '01**) Birds of a Feather: Homophily in Social Networks [[Paper](https://www.jstor.org/stable/2678628)]


### Link-level tasks (Link prediction)

* ![image](https://github.com/CurryTang/Towards-Graph-Foundation-Models-New-perspective-/assets/15672123/89a23a37-71d4-47f7-8949-7d859a41e369)(**ICLR '24**) Revisiting link prediction a data perspective [[Paper](https://arxiv.org/abs/2310.00793)]
* (**NIPS '23**) A Theory of Link Prediction via Relational Weisfeiler-Leman on Knowledge Graphs [[Paper](https://arxiv.org/abs/2302.02209)]
* (**ACL '23**) Are Message Passing Neural Networks Really Helpful for Knowledge Graph Completion? [[Paper](https://arxiv.org/abs/2205.10652)]
* 

### Graph-level tasks (Graph classification)

## GFM development: A data perspective


### Gathering more real-world data

We put a collection of large-scale real-world datasets below.

| Name                     	| URL                                                                      	| Description                                                                  	|
|--------------------------	|--------------------------------------------------------------------------	|------------------------------------------------------------------------------	|
| TU-Dataset               	| https://chrsmrrs.github.io/datasets/                                     	| A collection of graph-level prediction datasets                              	|
| NetworkRepository        	| https://networkrepository.com/                                           	| The largest graph datasets, with graphs coming from 30+ different domains    	|
| Open Graph Benchmark     	| https://ogb.stanford.edu/                                                	| Contains a bunch of large-scale graph datasets                               	|
| Pyg                      	| https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html 	| Official datasets provided by Pyg, containing popular datasets for benchmark 	|
| SNAP                     	| https://snap.stanford.edu/data/                                          	| Mainly focus on social network                                               	|
| Aminer                   	| https://www.aminer.cn/data/                                              	| A collection of academic graphs                                              	|
| OAG                      	| https://www.aminer.cn/open-academic-graph                                	| A large-scale academic graph                                                 	|
| MalNet                   	| https://www.mal-net.org/#home                                            	| A large-scale function calling graph for malware detection                   	|
| ScholKG                  	| https://scholkg.kmi.open.ac.uk/                                          	| A large-scale scholarly knowledge graph                                      	|
| Graphium                 	| https://github.com/datamol-io/graphium                                   	| A massive dataset for molecular property prediction                          	|
| Live Graph Lab           	| https://livegraphlab.github.io/                                          	| A large-scale temporal graph for NFT transactions                            	|
| Temporal Graph Benchmark 	| https://docs.tgb.complexdatalab.com/                                     	| A large-scale benchmark for temporal graph learning                          	|
| MoleculeNet              	| https://moleculenet.org/                                                 	| A benchmark for molecular machine learning                                   	|
| Recsys data              	| https://cseweb.ucsd.edu/~jmcauley/datasets.html                          	| A collection of datasets for recommender systems                             	|
| LINKX                    	| https://github.com/CUAI/Non-Homophily-Large-Scale                        	| A collection of large-scale non-homophilous graphs                           	|

### Handling feature heterogeneity



### Synthetic data generation
* (**KDD '22**) GraphWorld: Fake Graphs Bring Real Insights for GNNs [[Paper](https://arxiv.org/pdf/2203.00112.pdf)]

## GFM development: Backbone models

**A paper list of graph transformers [[Awesome graph transformers](https://github.com/wehos/awesome-graph-transformer)]**

* (**Arxiv '23**) GraphGPT: Graph Learning with Generative Pre-trained Transformers [[Paper](https://arxiv.org/abs/2401.00529)]
  

## GFM development: Training

**A paper list of self-supervised learning on graphs [[Awesome graph SSL](https://github.com/ChandlerBang/awesome-self-supervised-gnn)]**


## Existing GFM papers
* (**Arxiv '24**) A foundation model for atomistic materials chemistry [[Paper](https://arxiv.org/abs/2401.00096)]
* ![image](https://github.com/CurryTang/Towards-Graph-Foundation-Models-New-perspective-/assets/15672123/89a23a37-71d4-47f7-8949-7d859a41e369)(**ICLR '24**) Towards Foundation Models for Knowledge Graph Reasoning [[Paper](https://arxiv.org/abs/2310.04562)]
* (**ICLR '24**) One For All: Towards Training One Graph Model For All Classification Tasks [[Paper](https://openreview.net/forum?id=4IT2pgc9v6)]
* (**ICLR '24**) From Molecules to Materials: Pre-training Large Generalizable Models for Atomic Property Prediction [[Paper](https://arxiv.org/abs/2310.16802)]
* (**Arxiv '23**) DPA-2: Towards a universal large atomic model for molecular and material simulation [[Paper](https://arxiv.org/abs/2312.15492)]
* (**Arxiv '23**) GraphText: Graph Reasoning in Text Space [[Paper](https://arxiv.org/abs/2310.01089)]
* ![image](https://github.com/CurryTang/Towards-Graph-Foundation-Models-New-perspective-/assets/15672123/89a23a37-71d4-47f7-8949-7d859a41e369)(**NIPS '23**) PRODIGY: Enabling In-context Learning Over Graphs [[Paper](https://arxiv.org/abs/2305.12600)]
* (**Arxiv '23**) Towards Predicting Equilibrium Distributions for Molecular Systems with Deep Learning [[Paper](https://arxiv.org/abs/2306.05445)]

