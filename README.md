
## Hyperspectral datasets
  * Pavia University
  * Salinas
  * Indian Pines

An example dataset folder has the following structure:
```
Datasets
├── IndianPines
│   ├── Indian_pines_corrected.mat
│   ├── Indian_pines_gt.mat
├── Salinas
│   ├── salinas_gt.mat
│   └── salinas.mat
└── PaviaU
    ├── PaviaU_gt.mat
    └── PaviaU.mat
```

## Usage

Start a Visdom server:
`python -m visdom.server`
and go to [`http://localhost:8097`](http://localhost:8097) to see the visualizations 

Then, run the script `main.py`.

The most useful arguments are:
  * `--model` to specify the model (MVCH),
  * `--dataset` to specify which dataset to use (e.g. 'Indianpines', 'PaviaU', 'salinas'),
  * the `--cuda` switch to run the neural nets on GPU. The tool fallbacks on CPU if this switch is not specified.

Examples:
  * `python main.py --model MVCH --dataset PaviaU --training_sample 0.008 --cuda 0`


