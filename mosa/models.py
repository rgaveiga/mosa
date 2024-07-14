from typing import List, Tuple, Dict, Iterable, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np
from copy import deepcopy

class Group(BaseModel):
    '''
    name: string
        Name of the population group.
    data: list | tuple, optional
        Data that comprises the population group. If provided as a list, only 
        elements from this list can be selected for the tentative solution, resulting 
        in a discrete sample space. If provided as a tuple, the sample space is 
        continuous and bounded by the first and second elements of the tuple.
        The default value is (-1.0,1.0).
    change_value_move : float, optional
        Weight (non-normalized probability) used to select a trial move in which 
        the value of a randomly selected element in the solution will be modified. 
        How this modification is performed depends on the sample space of solutions 
        to the problem: (1) if discrete, values are exchanged between the solution 
        and the population; (2) if continuous, the value of an element in the 
        solution set is randomly incremented or decremented. The default value 
        is 1.0.
    insert_or_delete_move : float, optional
        Weight (non-normalized probability) used to select a trial move in which 
        an element will be inserted into or deleted from the solution set. The 
        default value is zero.
    swap_move : float, optional
        Weight (non-normalized probability) used to select a trial move in which 
        the algorithm swaps two randomly selected elements in the solution set. 
        The default value is zero.    
    mc_step_size : integer | float, optional
        Maximum Monte Carlo step size. The default value is None.
    number_of_solution_elements : integer, optional
        Number of solution elements. The default value is 1.
    maximum_number_of_solution_elements : integer, optional
        Maximum number of solution elements, if the number of elements is variable. 
        The default value is 10.
    element_repetition_allowed : boolean, optional
        Element can be repeated in the solution. The default value is True.
    sort_solution_elements : boolean, optional
        Solution elements are sorted in ascending order. The default value is 
        False.
    selection_weight : int | float, optional
        Group selection weight in a Monte Carlo iteration. The default value is 
        1.0.
    '''
    name: str
    data: List | Tuple = (-1.0,1.0)
    change_value_move: float = Field(1.0, ge=0.0)
    insert_or_delete_move: float = Field(0.0, ge=0.0)
    swap_move: float = Field(0.0, ge=0.0)
    mc_step_size: int | float | None = None
    number_of_solution_elements: int = Field(1, gt=0)
    maximum_number_of_solution_elements: int = Field(10, gt=0)
    element_repetition_allowed: bool = True
    sort_solution_elements: bool = False
    selection_weight: int | float = Field(1.0, gt=0.0)
    
    def append(self, value: Any) -> None:
        self.data.append(value)
        
    def pop(self, idx: int) -> None:
        self.data.pop(idx) 
    
    def remove(self, value: Any) -> None:
        self.data.remove(value)
        
    def copy(self):
        return deepcopy(self)
    
    def __iter__(self) -> Iterable:
        return iter(self.data)
    
    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]
             
    def __setitem__(self, idx: int, value: Any) -> None:
        self.data[idx] = value
             
    def __str__(self) -> str:
        data_str = str(self.data)
        
        if isinstance(self.data, Tuple):
            sample_space_str = " -> Continuous sample space"
        else:
            sample_space_str = " -> Discrete sample space"
        
        return data_str+sample_space_str
   
    @field_validator("name", mode="after")
    def validate_name(cls, name: str) -> str:
        if len(name)>0:
            return name
        else:
            raise ValueError("Name of a population group cannot be empty!")
             
    @field_validator("data", mode="after")
    def validate_data(cls, data: List | Tuple) -> List | Tuple:
        if len(data)>=2:
            if isinstance(data, List):
                return data
            elif isinstance(data, Tuple):
                if isinstance(data[0], (int,float)) and isinstance(data[1], (int,float)):
                    if data[1] > data[0]:                    
                        return data
                    else:
                        raise ValueError("Upper bound must be larger than lower bound!")
                else:
                    raise ValueError("Lower and upper bounds must be numbers!")
        else:
            raise RuntimeError("The group must contain at least two elements!")
                            
            
class Population(BaseModel):
    '''
    groups: list
        List of Group objects that make up the population.
    '''
    groups: List[Group]    
    
    def append(self, group: Group) -> None:
        self.groups.append(group)
        
    def pop(self, idx: int) -> None:
        self.groups.pop(idx)
    
    def remove(self, group: Group) -> None:
        self.groups.remove(group)
               
    def __iter__(self) -> Iterable:
        return iter(self.groups)
    
    def __getitem__(self, idx: int) -> Group:
        return self.groups[idx]
            
    def __setitem__(self, idx: int, group: Group) -> None:
        self.groups[idx] = group
        
    def __str__(self):
        s=""
        
        for group in self.groups:
            s+="- "+group.name+": "+str(group)+"\n"
            
        return s
            
    @field_validator("groups", mode="after")
    def validate_groups(cls, groups: List) -> List:
        if len(groups)>0:
            return groups
        else:
            raise RuntimeError("The population must comprise at least one group!")


class InputData(BaseModel):
    '''
    initial_temperature : float, optional
        Initial temperature for the Simulated Annealing algorithm. The default 
        value is 1.0.
    temperature_decrease_factor : float, optional
        Decrease factor of the temperature during Simulated Annealing. It
        determines how fast the quench will occur. The default value is 0.9.
    number_of_temperatures : integer, optional
        Number of temperatures. The default value is 10.
    number_of_iterations : integer, optional
        Number of Monte Carlo iterations per temperature. The default value is 
        1000.
    population : Population object
        A Population object from which data will be selected to create a tentative 
        solution to the problem.
    archive_file : string, optional
        Text file where the archive is saved. The default value is 'archive.json'.
    archive_size : integer, optional
        Maximum number of solutions in the archive. Default value is 1000.
    maximum_archive_rejections : integer, optional
        Maximum number of consecutive rejections of insertion of a solution 
        in the archive. Once reached, the optimization process finishes.
        Default value is 1000.
    alpha : float, optional
        Value of the alpha parameter. The default value is 0.0.
    restart : boolean, optional
        Specifies whether the optimization process should restart from a previous 
        run (if a checkpoint file is available) or not. The default value is True.
    track_optimization_progress : boolean, optional
        Track the optimization progress by saving the accepted objetive values 
        into a Python list. The default value is False.
    objective_weights : numpy array, optional
        Weights of the objectives. The default value is None, which implies the 
        same weight for all objectives.
    '''
    initial_temperature: float = Field(1.0, gt=0.0)
    temperature_decrease_factor: float = Field(0.9, gt=0.0)
    number_of_temperatures: int = Field(10, gt=0)
    number_of_iterations: int = Field(1000, gt=0)
    archive_file: str = "archive.json"
    population: Population
    archive_size: int = Field(1000, gt=0)
    maximum_archive_rejections: int = Field(1000, gt=0)
    alpha: float = Field(0.0, ge=0.0)
    restart: bool = True
    track_optimization_progress: bool = False
    objective_weights: np.ndarray | None = None
    
    @field_validator("archive_file", mode="after")
    def validate_archive_file(cls, archive_file: str) -> str:
        if len(archive_file)>0:
            return archive_file
        else:
            raise ValueError("The name of the archive file cannot be empty!")
    
            
class Solution(BaseModel):
    '''
    groups: list
        List of Group objects that make up the solution. The groups in the 
        solution must be the same as the groups in the population.
    x: dictionary, optional
        Dictionary where each key represents a group, and each value is the 
        corresponding data for that group. The dictionary should not be manually 
        assigned. The default value is {}.
    f: list, optional
        List of objective values for a given 'x'. The list should not be assigned 
        manually. The default value is [].
    '''
    group: List[Group]
    x: Dict = {}
    f: List[float] = []
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        # for group in self.groups:
        #     if isinstance(group, List):
        #         self.x[group.name] = list initialized
        #     else:
        #         self.x[group.name] = numpy array initialized
        
        #     setattr(self, group.name, self.x[group.name])
        
    def copy(self):
        return deepcopy(self)
    
    
class Archive(BaseModel):
    pass

    
    