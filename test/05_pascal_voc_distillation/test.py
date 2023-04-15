from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedopt import FedOpt
import unittest

if __name__ == "__main__":
    unittest.main()