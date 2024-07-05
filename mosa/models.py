from typing import List, Tuple

from pydantic import BaseModel, Field, field_validator

class Group(BaseModel):
    '''
    name: string
        Name of the population group.
    data: list | tuple, optional
        Data that comprises the population group. If provided as a list, only 
        elements from this list can be selected for the tentative solution, resulting 
        in a discrete sample space. If provided as a tuple, the sample space is 
        continuous and bounded by the first and second elements of the tuple.
        Default is (-1.0,1.0).
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
    '''
    name: str
    data: List | Tuple = (-1.0,1.0)
    change_value_move: float = Field(1.0, ge=0.0)
    insert_or_delete_move: float = Field(0.0, ge=0)
    swap_move: float = Field(0.0, ge=0.0)
    
    @field_validator("name")
    def validate_name(cls, name: str) -> str:
        if len(name)==0:
            raise ValueError("Name of a population group cannot be empty!")
            
class Population(BaseModel):
    _groups = List[Group]
    
    def append(self, group: Group) -> None:
        self._groups.append(group)

class InputData(BaseModel):
    '''
    initial_temperature : float, optional
        Initial temperature for the Simulated Annealing algorithm. The default 
        value is 1.0.
    temperature_decrease_factor : float, optional
        Decrease factor of the temperature during Simulated Annealing. It
        determines how fast the quench will occur. The default value is 0.9.
    population : Population object
        A Population object from which data will be selected to create a tentative 
        solution to the problem.
    restart : boolean, optional
        Specifies whether the optimization process should restart from a previous 
        run (if a checkpoint file is available) or not. The default is True.
    '''
    initial_temperature: float = Field(1.0, gt=0.0)
    temperature_decrease_factor: float = Field(0.9, gt=0.0)
    n_temp: int = Field(10, gt=0)
    population: Population
    restart: bool = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        for group in self.population._groups:
            setattr(self, group.name, group)
            
class Solution(BaseModel):
    pass

class Pareto(BaseModel):
    pass

    
    