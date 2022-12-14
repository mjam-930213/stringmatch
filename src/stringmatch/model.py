from __future__ import annotations
from typing import Tuple, List, Callable

import pandas as pd
import numpy as np

from Levenshtein import distance as levenshtein_distance
from Levenshtein import jaro as jaro_distance
from Levenshtein import jaro_winkler as jaro_winkler_distance
from fastDamerauLevenshtein import damerauLevenshtein

def check_algorithm_argument(algorithm: str) -> bool:
  """
  Checks if algorithm parameter is provided correctly.

  Args:
      algorithm: The algorithm used to train.

  Raises:
      Exception: Is algorithm a string?
      Exception: Is algorithm within accepted values?

  Returns:
      Bool.
  """
  if type(algorithm) != str:
    raise Exception('The type of algorithm must be string.')

  if algorithm not in ['levenshtein', 'damerau-levenshtein', 'jaro', 'jaro-winkler']:
    raise Exception('The algorithm provided is not one that is supported.')

  return True

def check_input_target_string_lists(input_string_list: str, target_string_list: str) -> bool:
  """
  Checks the input_string_list and target_string_list arguments.

  Args:
      input_string_list: A list of strings that needs matching.
      target_string_list: A list of strings that are possible match candidates.

  Returns:
      Bool.
  """
  if not all(isinstance(string, str) for string in input_string_list):
    raise Exception('All elements in input_string_list must be of type string.')

  if not all(isinstance(string, str) for string in target_string_list):
    raise Exception('All elements in target_string_list must be of type string.')
  
  return True

def calc_levenshtein_distance(input_string: str, target_string: str, weights: tuple = (1, 1, 1)) -> int:
  """
  Calculates the Levenshtein distance between two strings

  Args:
      input_string: The input string in which to calculate distance from
      target_string: The target string to which the input string is trying to match
      weights: The weights for the three operations in the form (insertion, 
               deletion, substitution). Default is (1, 1, 1), which gives all 
               three operations a weight of 1.

  Returns:
      Integer representing the number of operations (based on their weights) to transform
      from input_string to target_string (between 0 and inf)
  """
  # Type checks of input arguments
  if type(input_string) != str or type(target_string) != str:
    raise('input_string and target_string arguments must be of type str.')
   
  if len(weights) != 3:
    raise('Incorrect number of weights given - must be exactly 3')

  return levenshtein_distance(input_string, target_string, weights)

def calc_damerau_levenshtein_distance(input_string: str, target_string: str, weights: tuple = (1, 1, 1, 1)) -> int:
  """
  Calculates the Damerau-Levenshtein distance between two strings

  Args:
      input_string: The input string in which to calculate distance from
      target_string: The target string to which the input string is trying to match
      weights: The weights for the three operations in the form (insertion, 
               deletion, substitution, swapping). Default is (1, 1, 1, 1), which gives all 
               four operations a weight of 1. 

  Returns:
      Integer representing the number of operations (based on their weights) to transform
      from input_string to target_string (between 0 and inf).
  """
  # Type checks of input arguments
  if type(input_string) != str or type(target_string) != str:
    raise('input_string and target_string arguments must be of type str.')
   
  if len(weights) != 4:
    raise('Incorrect number of weights given - must be exactly 4')

  return damerauLevenshtein(input_string, 
                            target_string,
                            deleteWeight=weights[0],
                            insertWeight=weights[1],             
                            replaceWeight=weights[2],
                            swapWeight=weights[3],
                            similarity=False)

def calc_jaro_distance(input_string: str, target_string: str, weight: float = 1) -> float:
  """
  Calculates the Jaro similarity between two strings

  Args:
      input_string: The input string in which to calculate distance from
      target_string: The target string to which the input string is trying to match

  Returns:
      Float representing the similarity between the input_string to target_string, between 0 and 1.
      The closer to 0, the closer the two strings.
  """
  # Type checks of input arguments
  if type(input_string) != str or type(target_string) != str:
    raise('input_string and target_string arguments must be of type str.')

  return 1 - jaro_distance(input_string, target_string)

def calc_jaro_winkler_distance(input_string: str, target_string: str, weight: float = 0.1) -> float:
  """
  Calculates the Jaro-Winkler similarity between two strings

  Args:
      input_string: The input string in which to calculate distance from
      target_string: The target string to which the input string is trying to match
      weight: Weight used for the common prefix of the two strings. Defaults to 0.1.

  Returns:
      Float representing the similarity between the input_string to target_string, between 0 and 1.
      The closer to 0, the closer the two strings.
  """
  # Type checks of input arguments
  if type(input_string) != str or type(target_string) != str:
    raise('input_string and target_string arguments must be of type str.')
  
  if type(weight) != (float | int) or weight < 0 or weight > 1:
    raise('weight parameter is incorrect. Must be of type float or int, and between 0 and 1, inclusive.')

  return 1 - jaro_winkler_distance(input_string, target_string, prefix_weight=weight)

def calc_distance(input_string: str, target_string: str, algorithm: str = 'levenshtein', weights: (tuple | float) = (1, 1, 1)) -> (int | float):
  """
  Calculates the distance between two strings based on a set algorithm.

  Args:
      input_string: The input string in which to calculate distance from
      target_string: The target string to which the input string is trying to match
      algorithm: The algorithm used to calculate the distance between the two strings
      weight: Weight used for the specified algorithm. Could be a tuple of values or a float.

  Returns:
      Int or float representing the distance or similarity between the two strings.
  """
  # Type checks of input arguments
  if type(input_string) != str or type(target_string) != str or type(algorithm) != str:
    raise('input_string, target_string and algorithm arguments must be of type str.')
  
  if type(weights) != tuple and type(weights) != float:
    raise('weights argument must be of type tuple or float.')

  algorithms = {'levenshtein': calc_levenshtein_distance,
                'damerau-levenshtein': calc_damerau_levenshtein_distance,
                'jaro': calc_jaro_distance,
                'jaro-winkler': calc_jaro_winkler_distance
               }

  return algorithms[algorithm](input_string, target_string, weights)

def find_best_match_for_string(input_string: str, 
                               target_string_list: List[str], 
                               algorithm: str='levenshtein', 
                               weights: (tuple | float) = (1, 1, 1)) -> Tuple(str, (float | int)):
  """
  Find the best match out of a list of possible candidates for input_string using a specigic algorithm

  Args:
      input_string: The input string in which to calculate distance from
      target_string: The target string to which the input string is trying to match
      algorithm: The algorithm used to calculate the distance between the two strings
      weight: Weight used for the specified algorithm. Could be a tuple of values or a float.

  Returns:
      A tuple of the closest match to the input string and the score denoting how close 
      the match and the input string are.
  """
  
  # Type checks of input arguments
  if type(input_string) != str or type(algorithm) != str:
    raise('input_string, target_string and algorithm arguments must be of type str.')
  
  if not all(isinstance(string, str) for string in target_string_list):
    raise('All elements in target_string_list must be of type string.')

  if type(weights) != tuple and type(weights) != float:
    raise('weights argument must be of type tuple or float.')

  if algorithm in ['levenshtein', 'damerau-levenshtein', 'jaro', 'jaro-winkler']:
      best_estimate = ''
      best_distance = np.inf
      for target_string in target_string_list:
          distance = calc_distance(input_string, target_string, algorithm, weights)
          
          if distance < best_distance:
              best_estimate = target_string
              best_distance = distance
              
  else:
      raise('Invalid input into algorithm paramerter.')

  return best_estimate, best_distance
    
def find_best_match_for_list_of_strings(input_string_list: List[str],
                                        target_string_list: List[str],
                                        algorithm: str = 'levenshtein',
                                        weights: Tuple(int) = (1, 1, 1)) -> List[Tuple(str, str, (float | int))]:
  """
  Given a list of input strings find the best match for each input string 
  from a list of possible candidates.

  Args:
      input_string_list: List of input strings in which to calculate distance from
      target_string_list: List of target strings to which the input strings are matched against
      algorithm: The algorithm used to calculate the distance between the two strings
      weight: Weight used for the specified algorithm. Could be a tuple of values or a float.

  Returns:
      A list of tuples comprised of (input string, best match for the input strings, the distance between the two strings).
  """
  
  _ = check_input_target_string_lists(input_string_list, target_string_list)
  _ = check_algorithm_argument(algorithm)

  matches = []
  for input_string in input_string_list:
      best_estimate, best_distance = find_best_match_for_string(input_string, target_string_list, algorithm, weights)
      matches.append((input_string, best_estimate, best_distance))
      
  return matches

def calc_accuracy(y_pred: pd.Series, y_true: pd.Series) -> float:
  """
  Calculate accuracy between our predicted list of strings vs. groundtruth

  Args:
      y_pred: array-like of predicted candidates
      y_true: array-like of groundtruth target strings

  Returns:
      An accuracy for the number of "correct" predictions made.
  """
  if y_pred.shape[0] != y_true.shape[0]:
    raise('y_pred and y_true are not of the same-sized arrays.')

  return (y_true == y_pred).sum() / y_true.shape[0]

def calc_accuracy_for_matches(input_string_list: List[str],
                              target_string_list: List[str],
                              y_true: pd.Series,
                              algorithm: str, 
                              weights: (tuple | float) = (1, 1, 1)) -> float:
  """
  Finds a list of best matches for the list of input strings first. Then calculates the accuracy of the predictions.
  Args:
      input_string_list: List of input strings in which to calculate distance from
      target_string_list List of target strings to which the input strings are matched against
      y_true: array-like of groundtruth target strings
      algorithm: The algorithm used to calculate the distance between the two strings
      weights: Weight used for the specified algorithm. Could be a tuple of values or a float. Defaults to (1, 1, 1).

  Returns:
      Accuracy between the input list and target list of strings.
  """

  _ = check_input_target_string_lists(input_string_list, target_string_list)
  _ = check_algorithm_argument(algorithm)

  matches = find_best_match_for_list_of_strings(input_string_list, target_string_list, algorithm, weights= weights)
  df_best_matches = pd.DataFrame(matches, columns=['Input', 'Estimate', 'Distance'])
  y_pred = df_best_matches['Estimate']
  accuracy = calc_accuracy(y_pred, y_true)

  return accuracy

def cross_val_levenshtein(input_string_list: List[str], 
                          target_string_list: List[str], 
                          y_true: pd.Series,
                          algorithm: str,
                          max_weights: Tuple(int, int, int), 
                          verbose: bool = True) -> List[Tuple(Tuple(int, int, int), float)]:
  """
  Cross validate to find the best parameters for the Levenshtein algorithm.

  Args:
      input_string_list: List of input strings in which to calculate distance from
      target_string_list: List of target strings to which the input strings are matched against
      y_true: array-like of groundtruth target strings
      algorithm: The algorithm used to calculate the distance between the two strings
      max_weights: Maximum value/values the chosen algorithm should use.
      verbose: Flag to indicate whether to display training progress. Defaults to True.

  Returns:
      List tuples comprised of a tuple of the weights and the accuracy associated with that set of weights.
  """
  _ = check_input_target_string_lists(input_string_list, target_string_list)
  _ = check_algorithm_argument(algorithm)

  weights_score = []
  for i in range(1, max_weights[0]+1):
          for j in range(1, max_weights[1]+1):
              for k in range(1, max_weights[2]+1):
                  accuracy = calc_accuracy_for_matches(input_string_list, target_string_list, y_true, algorithm, max_weights)
                  weights_score.append(((i, j, k), accuracy))

                  if verbose:
                      print('({}, {}, {}): {:.1%}'.format(i, j, k, accuracy))
  
  return weights_score

def cross_val_damerau_levenshtein(input_string_list: List[str], 
                                  target_string_list: List[str], 
                                  y_true: pd.Series,
                                  algorithm: str,
                                  max_weights: Tuple(int, int, int, int), 
                                  verbose: bool = True) -> List[Tuple(Tuple(int, int, int, int), float)]:
  """
  Cross validate to find the best parameters for the Damerau-Levenshtein algorithm.

  Args:
      input_string_list: List of input strings in which to calculate distance from
      target_string_list: List of target strings to which the input strings are matched against
      y_true: array-like of groundtruth target strings
      algorithm: The algorithm used to calculate the distance between the two strings
      max_weights: Maximum value/values the chosen algorithm should use. Could be a tuple of values or a float.
      verbose: Flag to indicate whether to display training progress. Defaults to True.

  Returns:
      List tuples comprised of a tuple of the weights and the accuracy associated with that set of weights.
  """

  _ = check_input_target_string_lists(input_string_list, target_string_list)
  _ = check_algorithm_argument(algorithm)

  weights_score = []
  for i in range(1, max_weights[0]+1):
          for j in range(1, max_weights[1]+1):
              for k in range(1, max_weights[2]+1):
                  for m in range(1, max_weights[3]+1):
                      accuracy = calc_accuracy_for_matches(input_string_list, target_string_list, y_true, algorithm, max_weights)
                      weights_score.append(((i, j, k, m), accuracy))

                      if verbose:
                          print('({}, {}, {}, {}): {:.1%}'.format(i, j, k, m, accuracy))

  return weights_score

def cross_val_jaro(input_string_list: List[str], 
                    target_string_list: List[str], 
                    y_true: pd.Series,
                    algorithm: str,
                    max_weights: float = 0.0,
                    verbose: bool = True) -> List[float]:
  """
  Cross validate to find the best parameters for the Jaro algorithm.

  Args:
      input_string_list: List of input strings in which to calculate distance from
      target_string_list: List of target strings to which the input strings are matched against
      y_true: array-like of groundtruth target strings
      algorithm: The algorithm used to calculate the distance between the two strings
      max_weights: Not actually used, but included so the code will work.
      verbose: Flag to indicate whether to display training progress. Defaults to True.

  Returns:
      List floats denoting accuracy.
  """

  _ = check_input_target_string_lists(input_string_list, target_string_list)
  _ = check_algorithm_argument(algorithm)

  accuracy = calc_accuracy_for_matches(input_string_list, target_string_list, y_true, algorithm, max_weights)
  weights_score = accuracy

  if verbose:
      print('{:.1%}'.format(accuracy))

  return weights_score

def cross_val_jaro_winkler(input_string_list: List[str], 
                          target_string_list: List[str], 
                          y_true: pd.Series,
                          algorithm: str,
                          weights: np.ndarray = np.linspace(0, 1, 10),
                          verbose: bool = True) -> List[Tuple(float, float)]:
  """
  Cross validate to find the best parameters for the Jaro-Winkler algorithm.

  Args:
      input_string_list: List of input strings in which to calculate distance from
      target_string_list: List of target strings to which the input strings are matched against
      y_true: array-like of groundtruth target strings
      algorithm: The algorithm used to calculate the distance between the two strings
      weights: The list of weights used to search for best parameters.
      verbose: Flag to indicate whether to display training progress. Defaults to True.

  Returns:
      List tuples comprised of the weight and its corresponding accuracy.
  """

  _ = check_input_target_string_lists(input_string_list, target_string_list)
  _ = check_algorithm_argument(algorithm)

  weights_score = []
  for i in weights:
    accuracy = calc_accuracy_for_matches(input_string_list, target_string_list, y_true, algorithm, weights)
    weights_score.append((i, accuracy))

    if verbose:
        print('{}: {:.1%}'.format(i, accuracy))
  
  return weights_score

def cross_val_weights(input_string_list, target_string_list, y_true, algorithm='levenshtein', max_weights=(1, 1, 1), verbose=False):
  """
  Cross validate to find the best parameters for the Damerau-Levenshtein algorithm.

  Args:
      input_string_list: List of input strings in which to calculate distance from
      target_string_list: List of target strings to which the input strings are matched against
      y_true: array-like of groundtruth target strings
      algorithm: The algorithm used to calculate the distance between the two strings
      max_weights: Maximum value/values the chosen algorithm should use. Could be a tuple of values or a float.
      verbose: Flag to indicate whether to display training progress. Defaults to True.

  Returns:
      List tuples comprised of a tuple of the weights and the accuracy associated with that set of weights.
  """

  _ = check_input_target_string_lists(input_string_list, target_string_list)
  _ = check_algorithm_argument(algorithm)

  algorithms = {'levenshtein': cross_val_levenshtein,
                'damerau-levenshtein': cross_val_damerau_levenshtein,
                'jaro': cross_val_jaro,
                'jaro-winkler': cross_val_jaro_winkler
               }
  
  weights_score = algorithms[algorithm](input_string_list, target_string_list, y_true, algorithm, max_weights, verbose)
              
  return weights_score
