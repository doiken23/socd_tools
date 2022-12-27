# SOCD: Synthesized Object-Level Change Detection Dataset

This is a codebase for **SOCD: Synthesized Object-Level Change Detection Dataset**.
Codes for data loading, visualization, and evaluation are provided.

- [Paper](https://www.mdpi.com/2072-4292/14/17/4225)
- [Project page](https://doiken23.github.io/object_level_cd/dataset.html)


## Contents

- `cocoobjcdapi/`: library for evaluation.
- `dataset/`: data loading and visualization tools.
- `evaluation/`: evaluation tools.


## Installation

```
pip install -r requirements.txt
```

If you would like use evaluation tools.

```
cd cocoobjcdapi/
make
```


## Citation

If you find our dataset helpful, please cite the paper:
```
@article{objcd,
  author    = {Doi, Kento and Hamaguchi, Ryuhei and Iwasawa, Yusuke and Onishi, Masaki and Matsuo, Yutaka and Sakurada, Ken},
  title     = {Detecting Object-Level Scene Changes in Images with Viewpoint Differences Using Graph Matching},
  journal   = {Remote Sensing},
  volume    = {14},
  number    = {17},
  year      = {2022},
}
```
