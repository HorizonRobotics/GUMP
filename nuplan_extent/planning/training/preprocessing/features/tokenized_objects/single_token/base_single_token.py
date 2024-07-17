from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from abc import ABC, abstractmethod

class BaseSingleToken(ABC):
    """
    Abstract base class for a single token in a tokenized transformer model.

    This class provides a template for tokens with common attributes and methods that 
    can be inherited by specific types of tokens, ensuring consistency and the 
    possibility of polymorphism when handling different types of tokens in the model.

    Attributes:
        value (str): The string representation of the token.
        token_type (str): A string that categorizes the token into a type, 
                          e.g., 'control', 'agent', etc.
    """

    def __init__(self, value, frame_index, token_type):
        """
        Initializes the BaseSingleToken instance with a value and a token type.

        Args:
            value (str): The string representation of the token.
            token_type (str): A categorization of the token.
        """
        self.value = value
        self.frame_index = frame_index
        self.token_type = token_type
        
        # position in the frame
        self.index_in_frame = None
        self.index_in_sequence = None
        self.valid_for_training = True
        self.is_imagined = False
    
    @abstractmethod
    def name(self):
        """
        Abstract method to get the name or identifier of the token.

        This method should be implemented in subclasses to return a unique name or 
        identifier for the token. The name can be used for identifying the token type 
        in processing, logging, debugging, or other operations.

        Returns:
            str: A string representing the unique name or identifier of the token.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def abbreviation(self):
        """
        Abstract method to get the abbreviation of the token.

        This method should be implemented in subclasses to return a unique abbreviation 
        for the token. The abbreviation can be used for identifying the token type in 
        processing, logging, debugging, or other operations.

        Returns:
            str: A string representing the unique abbreviation of the token.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def __str__(self):
        """
        Returns a string representation of the token.

        Overrides the default string representation method to provide a human-readable
        description of the token, including its value and type.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def next_possible_token_type(self):
        """
        Abstract method to get the next possible tokens in the sequence.

        This method should be implemented in subclasses to return a list of the next 
        possible tokens in the sequence. The list of next possible tokens can be used 
        for identifying the token type in processing, logging, debugging, or other 
        operations.

        Returns:
            List[BaseSingleToken]: A list of the next possible tokens in the sequence.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def assign_position(self, index_in_sequence):
        """
        Assigns the position of the token in the tokenized sequence.
        """
        self.index_in_sequence = index_in_sequence
    
