# parallel_data_processing_engine
### Ideal Project Structure
```
correlation-parallel-project/

GPU_CPU.docx
LICENSE
Project_descriptions
README.md
test_cases/

src/
    cpu/
        dataset.py
        serial_cpu.py
        parallel_cpu.py
        benchmark.py
        visualize.py


    gpu/        <-- for the other person later
        gpu_correlation.py

run_experiment.py
results/

```
## Some important Instructions
```
Shared experiment runner

Keep this in root:

run_experiment.py


This script can run both:

CPU benchmark
GPU benchmark


Later you could even do:

python run_experiment.py --mode cpu ## default run:: python run_experiment.py
python run_experiment.py --mode gpu

Results folder

Create:

results/


Save:

runtime_plot.png
correlation_heatmap.png
benchmark_table.csv


This is very useful for the report.
```

### GPU computing can to be done in google collab. Everyone won't have a dedicated GPU and even if a dedicated GPU is available there are many dependencies required for CuPy or PyTorch that may take up quiet some time to install locally, so best idea would be to use google collab.
