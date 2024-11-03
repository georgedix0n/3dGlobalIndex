# CUDA 3D Unique GID Calculation Example

This repository provides a CUDA example that demonstrates **unique global ID (GID) calculation** in a 3D grid and block structure.

## Overview

The program contains a kernel function, `unique_gid_calculation_3d`, which calculates a unique global ID (GID) for each thread in a 3D block and grid configuration. Each thread uses its GID to access specific data in a device array, printing relevant thread and block information.

## Execution

The kernel is launched with a 3D grid and block configuration, and a sample dataset is copied from the host to the device for demonstration.

