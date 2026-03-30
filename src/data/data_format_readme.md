"""
Data Format Specification

Core Requirement:
All datasets must follow a strict format representing N time series with T time steps,
such that the data can be converted into a matrix with shape:

    X.shape = (N, T)

--------------------------------------------------

File Format (Mandatory):

#Series1,Series2,Series3,...,SeriesN
x11,x12,x13,...,x1N
x21,x22,x23,...,x2N
x31,x32,x33,...,x3N
...
xT1,xT2,xT3,...,xTN

--------------------------------------------------

Interpretation:

- First line:
  - Must start with '#'
  - Treated as metadata/header
  - Contains series names (optional, ignored by loader)

- Remaining lines:
  - Must contain only numeric values
  - Each row represents one time step
  - Each column represents one time series

--------------------------------------------------

Internal Representation:

- Raw file shape: (T, N)
- Transformed to: (N, T)

Where:
- N = number of time series
- T = number of time steps

--------------------------------------------------

Example Dataset (Realistic):

#Amsterdam,Berlin,Delhi
2.57,1.01,15.08
6.66,2.73,15.76
4.75,1.74,14.41
4.13,1.29,13.53

Interpretation:
- N = 3
- T = 4

After loading:
    X.shape = (3, 4)

--------------------------------------------------

Example Synthetic Dataset:

#S1,S2,S3
0.12,-0.45,1.03
-1.22,0.33,0.56
0.78,-0.91,-0.12
0.44,0.67,-0.89

After loading:
    X.shape = (3, 4)

--------------------------------------------------

Strict Rules:

- First line must start with '#'
- No text or strings in data rows
- All values must be numeric
- All rows must have the same number of columns
- No missing values (NaN)

--------------------------------------------------

Invalid Examples:

1. Missing header marker:

Series1,Series2
1,2
3,4

2. Non-numeric values:

#S1,S2
1,2
a,4

3. Inconsistent columns:

#S1,S2,S3
1,2
3,4,5

--------------------------------------------------

Summary:

- Header: Required, must start with '#'
- Data type: Numeric only
- File shape: (T, N)
- Internal shape: (N, T)

--------------------------------------------------

Notes:

- Header names are ignored during computation
- Normalization is handled within compute functions
- This format is required for both CPU and GPU pipelines
"""

