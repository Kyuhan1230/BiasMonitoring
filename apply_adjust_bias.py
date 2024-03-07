import numpy as np


class BiasAdjustment:
    """
    A class to gradually adjust bias values in real time. 
    The bias value can change gradually ('gradual') or linearly ('linear') until it reaches the target value.

    Attributes:
        response_time (int): The number of indexes required for the bias to complete its change.
        bias_level (float): The current bias value.
        target_bias (float): The bias value aimed to be reached.
        last_target_bias (float): The previous target bias value before the current target.
        target_index (int): The index by which the target bias should be achieved.
        last_update_index (int): The last index at which the bias was updated.

    Methods:
        validate_bias_input(index, bias_input): Validates if a new bias input is different from the current target.
        update_target_bias(new_index, new_bias): Updates the target bias and when it should be achieved.
        calculate_bias(current_index, bias_type): Calculates the new bias based on the bias type and current index.
        apply_bias(index, bias_input, bias_type): Applies the new bias if it is validated and returns the new bias level.
    """
    def __init__(self, response_time=20, bias_init=0):
        """Initializes a new RealTimeBiasAdjustment object with default or provided values.
        
        Args:
            - response_time (int): The number of index units it takes for the bias to fully adjust to a new target.
            - bias_init (float): The initial bias level to start from.
        """
        self.response_time = response_time
        self.bias_level = bias_init
        self.target_bias = 0
        self.last_target_bias = bias_init
        self.target_index = 0
        self.last_update_index = 0

    def validate_bias_input(self, index, bias_input):
        """Validates the incoming bias input against the current target bias.
        
        Args:
            - index (int): The current index at which the bias input is provided.
            - bias_input (float): The new bias value proposed by the system.

        Returns:
            - bool: True if the new bias input is different from the current target bias, False otherwise.
        """
        if self.target_bias != bias_input:
            self.update_target_bias(index, bias_input)
            return True
        return False

    def update_target_bias(self, new_index, new_bias):
        """Updates the bias settings for a new target.
        
        Args:
            - new_index (int): The current index at which the new target bias is set.
            - new_bias (float): The new target bias value to be achieved.
        """
        # Update the previous target bias to reflect the last known target before updating to a new one.
        self.last_target_bias = self.target_bias

        self.target_bias = new_bias
        self.target_index = new_index + self.response_time
        self.last_update_index = new_index
        
    def calculate_bias(self, current_index, bias_type='gradual'):
        """Calculates the new bias based on the type and current index.
        
        Args:
            - current_index (int): The current step or index in the adjustment process.
            - bias_type (str): The type of bias adjustment ('gradual' or 'linear').

        Returns:
            - float: The calculated bias level at the current index.
        """
        time_elapsed = current_index - self.last_update_index
        if bias_type == 'gradual':
            # 'Gradual' bias type calculations
            # Step Function Response = K × (1 - exp(-t/τ)) ; (K: final value, t: time, τ: time constant)
            if current_index < self.target_index:
                change = (1 - np.exp(-time_elapsed / self.response_time))
                bias_level = self.bias_level + (self.target_bias - self.bias_level) * change
            else:
                bias_level = self.target_bias  # target_index에 도달하면 target_bias로 설정
        
        elif bias_type == 'linear':
            # 'Linear' bias type calculations
            if current_index < self.target_index:
                slope = (self.target_bias - self.last_target_bias) / (self.target_index - self.last_update_index)
                bias_level = slope * (current_index - self.last_update_index) + self.last_target_bias
            else:
                bias_level = self.target_bias
        else:
            # Default case: No change in bias
            bias_level = self.bias_level

        self.bias_level = bias_level
        return bias_level

    def apply_bias(self, index, bias_input, bias_type='gradual'):
        """Applies bias based on the type, index, and input value.
        
        Args:
            - index (int): The current index at which the bias input is being applied.
            - bias_input (float): The new bias value proposed by the system.
            - bias_type (str): The type of bias adjustment to apply ('gradual' or 'linear').

        Returns:
            - float: The newly calculated bias level after application.
        """
        is_changed = self.validate_bias_input(index=index, bias_input=bias_input)
        has_not_arrived = index <= self.target_index

        if is_changed or has_not_arrived:
            return self.calculate_bias(index, bias_type)
        return self.bias_level
