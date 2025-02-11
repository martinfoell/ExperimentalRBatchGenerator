from __future__ import annotations
import ROOT
ROOT.gInterpreter.Declare('#include "../inc/RBatchGenerator_python.hxx"')
from typing import Any, Callable, Tuple, TYPE_CHECKING
import atexit

if TYPE_CHECKING:
    import numpy as np
    import tensorflow as tf
    import torch



class BaseGenerator:
    def get_template(
        self,
        x_rdf: RNode,
        columns: list[str] = list(),
    ) -> Tuple[str, list[int]]:
        """
        Generate a template for the RBatchGenerator based on the given
        RDataFrame and columns.

        Args:
            rdataframe (RNode): RDataFrame or RNode object.
            columns (list[str]): Columns that should be loaded.
                                 Defaults to loading all columns
                                 in the given RDataFrame
        Returns:
            template (str): Template for the RBatchGenerator
        """

        if not columns:
            columns = x_rdf.GetColumnNames()

        template_string = ""

        self.given_columns = []
        self.all_columns = []

        max_vec_sizes_list = []

        for name in columns:
            name_str = str(name)
            self.given_columns.append(name_str)
            column_type = x_rdf.GetColumnType(name_str)
            template_string = f"{template_string}{column_type},"
            self.all_columns.append(name_str)

        return template_string[:-1]

    def __init__(
        self,
        rdataframe: RNode,
        num_epochs: int,                
        chunk_size: int,        
        range_size: int,                    
        batch_size: int,
        columns: list[str] = list(),
        target: str | list[str] = list(),
        validation_split: float = 0,
        shuffle: bool = True,
    ):
        """Wrapper around the Cpp RBatchGenerator

            Args:
            rdataframe (RNode): Name of RNode object.
            batch_size (int): Size of the returned chunks.
            chunk_size (int):
                The size of the chunks loaded from the ROOT file. Higher chunk size
                results in better randomization, but also higher memory usage.
            columns (list[str], optional):
                Columns to be returned. If not given, all columns are used.
            max_vec_sizes (dict[std, int], optional):
                Size of each column that consists of vectors.
                Required when using vector based columns.
            vec_padding (int):
                Value to pad vectors with if the vector is smaller
                than the given max vector length. Defaults is 0
            target (str|list[str], optional):
                Column(s) used as target.
            weights (str, optional):
                Column used to weight events.
                Can only be used when a target is given.
            validation_split (float, optional):
                The ratio of batches being kept for validation.
                Value has to be between 0 and 1. Defaults to 0.0.
            max_chunks (int, optional):
                The number of chunks that should be loaded for an epoch.
                If not given, the whole file is used.
            shuffle (bool):
                Batches consist of random events and are shuffled every epoch.
                Defaults to True.
            drop_remainder (bool):
                Drop the remainder of data that is too small to compose full batch.
                Defaults to True.
        """

        import ROOT
        from ROOT import RDF

        try:
            import numpy as np

        except ImportError:
            raise ImportError(
                "Failed to import NumPy during init. NumPy is required when \
                    using RBatchGenerator"
            )

        if chunk_size < batch_size:
            raise ValueError(
                f"chunk_size cannot be smaller than batch_size: chunk_size: \
                    {chunk_size}, batch_size: {batch_size}"
            )

        if validation_split < 0.0 or validation_split > 1.0:
            raise ValueError(
                f"The validation_split has to be in range [0.0, 1.0] \n \
                    given value is {validation_split}"
            )

        self.noded_rdf = RDF.AsRNode(rdataframe)

        if ROOT.Internal.RDF.GetDataSourceLabel(self.noded_rdf) != "TTreeDS":
            raise ValueError(
                "RNode object must be created out of TTrees or files of TTree"
            )

        if isinstance(target, str):
            target = [target]

        self.target_columns = target

        template = self.get_template(
            rdataframe, columns
        )

        self.num_columns = len(self.all_columns)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Handle target
        self.target_given = len(self.target_columns) > 0
        # self.weights_given = len(self.weights_column) > 0
        if self.target_given:
            for target in self.target_columns:
                if target not in self.all_columns:
                    raise ValueError(
                        f"Provided target not in given columns: \ntarget => \
                            {target}\ncolumns => {self.all_columns}")

            self.target_indices = [self.all_columns.index(
                target) for target in self.target_columns]

            self.train_indices = [c for c in range(
                    len(self.all_columns)) if c not in self.target_indices]

        else:
            self.train_indices = [c for c in range(len(self.all_columns))]

        self.train_columns = [
            c for c in self.all_columns if c not in self.target_columns]

        from ROOT import EnableThreadSafety

        # The RBatchGenerator will create a separate C++ thread for I/O.
        # Enable thread safety in ROOT from here, to make sure there is no
        # interference between the main Python thread (which might call into
        # cling via cppyy) and the I/O thread.
        EnableThreadSafety()

        self.generator = ROOT.RBatchGenerator(template)(
            self.noded_rdf,
            num_epochs,
            chunk_size,
            range_size,
            batch_size,
            validation_split,            
            shuffle,            
            self.given_columns,
        )

        # atexit.register(self.DeActivate)

    @property
    def is_active(self):
        return self.generator.IsActive()

    def Activate(self):
        """Initialize the generator to be used for a loop"""
        self.generator.Activate()

    def DeActivate(self):
        """Deactivate the generator"""
        self.generator.DeActivate()

    def ActivateEpoch(self):
        """Start the loading of training batches"""
        self.generator.ActivateEpoch()

    def DeActivateEpoch(self):
        """Stop the loading of batches"""

        self.generator.DeActivateEpoch()
        


    def ConvertBatchToPyTorch(self, batch: Any) -> torch.Tensor:
        """Convert a RTensor into a PyTorch tensor

        Args:
            batch (RTensor): Batch returned from the RBatchGenerator

        Returns:
            torch.Tensor: converted batch
        """
        import torch
        import numpy as np

        data = batch.GetData()
        batch_size, num_columns = tuple(batch.GetShape())

        data.reshape((batch_size * num_columns,))

        return_data = torch.as_tensor(np.asarray(data)).reshape(
            batch_size, num_columns)

        # Splice target column from the data if target is given
        if self.target_given:
            train_data = return_data[:, self.train_indices]
            target_data = return_data[:, self.target_indices]

            if len(self.target_indices) == 1:
                return train_data, target_data.reshape(-1, 1)

            return train_data, target_data

        return return_data

    # Return a batch when available
    def GetTrainBatch(self) -> Any:
        """Return the next training batch of data from the given RDataFrame

        Returns:
            (np.ndarray): Batch of data of size.
        """

        batch = self.generator.GetTrainBatch()

        if batch and batch.GetSize() > 0:
            return batch

        return None

    # Return a batch when available
    
    def GetValidationBatch(self) -> Any:
        """Return the next training batch of data from the given RDataFrame

        Returns:
            (np.ndarray): Batch of data of size.
        """

        batch = self.generator.GetValidationBatch()

        if batch and batch.GetSize() > 0:
            return batch

        return None
    
# Context that activates and deactivates the loading thread of the Cpp class
# This ensures that the thread will always be deleted properly
class LoadingThreadContext:
    def __init__(self, base_generator: BaseGenerator):
        self.base_generator = base_generator

    def __enter__(self):
        self.base_generator.Activate()

    def __exit__(self, type, value, traceback):
        self.base_generator.DeActivate()
        return True


class TrainRBatchGenerator:

    def __init__(self, base_generator: BaseGenerator, conversion_function: Callable):
        """
        A generator that returns the training batches of the given
        base generator

        Args:
            base_generator (BaseGenerator):
                The base connection to the Cpp code
            conversion_function (Callable[RTensor, np.NDArray|torch.Tensor]):
                Function that converts a given RTensor into a python batch
        """
        self.base_generator = base_generator
        self.conversion_function = conversion_function

    def Activate(self):
        """Start the loading of training batches"""
        self.base_generator.Activate()

    def DeActivate(self):
        """Stop the loading of batches"""

        self.base_generator.DeActivate()

        
    @property
    def columns(self) -> list[str]:
        return self.base_generator.all_columns

    @property
    def train_columns(self) -> list[str]:
        return self.base_generator.train_columns

    @property
    def target_columns(self) -> str:
        return self.base_generator.target_columns

    @property
    def weights_column(self) -> str:
        return self.base_generator.weights_column

    @property
    def number_of_batches(self) -> int:
        return self.base_generator.generator.NumberOfTrainingBatches()

    @property
    def last_batch_no_of_rows(self) -> int:
        return self.base_generator.generator.TrainRemainderRows()

    def __iter__(self):
        self._callable = self.__call__()

        return self

    def __next__(self):
        batch = self._callable.__next__()

        if batch is None:
            raise StopIteration

        return batch

    def __call__(self) -> Any:
        """Start the loading of batches and Yield the results

        Yields:
            Union[np.NDArray, torch.Tensor]: A batch of data
        """
        # self.base_generator.ActivateEpoch()
        # ActivateEpoch()        
        print("Testing")
        # with LoadingThreadContext(self.base_generator):
        while True:
            batch = self.base_generator.GetTrainBatch()
            # self.base_generator.DeActivateEpoch()                                    
            if batch is None:
                self.base_generator.DeActivateEpoch()                    
                print("No more batches")
                break
            yield self.conversion_function(batch)
        
        return None

    # def __call__(self) -> Any:
    #     """Start the loading of batches and Yield the results

    #     Yields:
    #         Union[np.NDArray, torch.Tensor]: A batch of data
    #     """

    #     with LoadingThreadContext(self.base_generator):
    #         while True:
    #             batch = self.base_generator.GetTrainBatch()

    #             if batch is None:
    #                 break

    #             yield self.conversion_function(batch)

    #     return None
    

class ValidationRBatchGenerator:
    def __init__(self, base_generator: BaseGenerator, conversion_function: Callable):
        """
        A generator that returns the validation batches of the given base
        generator. NOTE: The ValidationRBatchGenerator only returns batches
        if the training has been run.

        Args:
            base_generator (BaseGenerator):
                The base connection to the Cpp code
            conversion_function (Callable[RTensor, np.NDArray|torch.Tensor]):
                Function that converts a given RTensor into a python batch
        """
        self.base_generator = base_generator
        self.conversion_function = conversion_function

    @property
    def columns(self) -> list[str]:
        return self.base_generator.all_columns

    @property
    def train_columns(self) -> list[str]:
        return self.base_generator.train_columns

    @property
    def target_columns(self) -> str:
        return self.base_generator.target_columns

    @property
    def weights_column(self) -> str:
        return self.base_generator.weights_column

    @property
    def number_of_batches(self) -> int:
        return self.base_generator.generator.NumberOfValidationBatches()

    @property
    def last_batch_no_of_rows(self) -> int:
        return self.base_generator.generator.ValidationRemainderRows()

    def __iter__(self):
        self._callable = self.__call__()

        return self

    def __next__(self):
        batch = self._callable.__next__()

        if batch is None:
            raise StopIteration

        return batch

    def __call__(self) -> Any:
        """Loop through the validation batches

        Yields:
            Union[np.NDArray, torch.Tensor]: A batch of data
        """
        if self.base_generator.is_active:
            self.base_generator.DeActivate()

        while True:
            batch = self.base_generator.GetValidationBatch()

            if not batch:
                break

            yield self.conversion_function(batch)


def CreatePyTorchGenerators(
    rdataframe: RNode,
    num_epochs: int,            
    chunk_size: int,        
    range_size: int,                    
    batch_size: int,
    columns: list[str] = list(),
    target: str | list[str] = list(),
    validation_split: float = 0,
    shuffle: bool = True,
) -> Tuple[TrainRBatchGenerator, ValidationRBatchGenerator]:
    """
    Return two Tensorflow Datasets based on the given ROOT file and tree or RDataFrame
    The first generator returns training batches, while the second generator
    returns validation batches

    Args:
        rdataframe (RNode): Name of RNode object.
        batch_size (int): Size of the returned chunks.
        chunk_size (int):
            The size of the chunks loaded from the ROOT file. Higher chunk size
            results in better randomization, but also higher memory usage.
        columns (list[str], optional):
            Columns to be returned. If not given, all columns are used.
        max_vec_sizes (list[int], optional):
            Size of each column that consists of vectors.
            Required when using vector based columns
        target (str|list[str], optional):
            Column(s) used as target.
        weights (str, optional):
            Column used to weight events.
            Can only be used when a target is given
        validation_split (float, optional):
            The ratio of batches being kept for validation.
            Value has to be from 0.0 to 1.0. Defaults to 0.0.
        max_chunks (int, optional):
            The number of chunks that should be loaded for an epoch.
            If not given, the whole file is used
        shuffle (bool):
            randomize the training batches every epoch.
            Defaults to True
        drop_remainder (bool):
            Drop the remainder of data that is too small to compose full batch.
            Defaults to True.
            Let a data list [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] with batch_size=4 be
            given.
            If drop_remainder = True, then two batches [0, 1, 2, 3] and
            [4, 5, 6, 7] will be returned.
            If drop_remainder = False, then three batches [0, 1, 2, 3],
            [4, 5, 6, 7] and [8, 9] will be returned.

    Returns:
        TrainRBatchGenerator or
            Tuple[TrainRBatchGenerator, ValidationRBatchGenerator]:
            If validation split is 0, return TrainBatchGenerator.

            Otherwise two generators are returned. One used to load training
            batches, and one to load validation batches. NOTE: the validation
            batches are loaded during the training. Before training, the
            validation generator will return no batches.
    """
    base_generator = BaseGenerator(
        rdataframe,
        num_epochs,
        chunk_size,
        range_size,
        batch_size,
        columns,
        target,
        validation_split,
        shuffle,
    )

    train_generator = TrainRBatchGenerator(
        base_generator, base_generator.ConvertBatchToPyTorch
    )

    if validation_split == 0.0:
        return train_generator

    # validation_generator = ValidationRBatchGenerator(
    #     base_generator, base_generator.ConvertBatchToPyTorch
    # )
    
    validation_generator = 0
    
    

    return train_generator, validation_generator
