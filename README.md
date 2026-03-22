# cda_case1


### Observations
C2 only has 72 as a value, so it's really useless

### Idea
The basic idea is ofc that we do not need all the variables, only a handful of variables are going to carry meaningful data

for this, first, we need to get which columns are the most correlated to the y column, which is the prediction column.

```c = df.corr()['y'].abs().sort_values(ascending=False)```

this gives:

```
x_32    0.770303
x_36    0.478040
x_31    0.426011
x_14    0.404301
x_94    0.330637
x_40    0.325519
x_55    0.323713
x_28    0.313412
x_93    0.303670
x_11    0.302154
x_73    0.296177
x_04    0.293943
x_42    0.281429
x_58    0.278323
x_84    0.277369
x_37    0.276967
x_67    0.268328
x_62    0.267468
x_95    0.266938
```

for as the top 20 most correlated columns to `y`