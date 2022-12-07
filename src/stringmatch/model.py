from __future__ import annotations
from typing import Tuple, List

import pandas as pd
import numpy as np

from Levenshtein import distance as levenshtein_distance
from Levenshtein import jaro as jaro_distance
from Levenshtein import jaro_winkler as jaro_winkler_distance
from fastDamerauLevenshtein import damerauLevenshtein


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

  if type(input_string) != str or type(target_string) != str:
    raise('input_string and target_string arguments must be of type str.')
   
  if len(weights) != 4:
    raise('Incorrect number of weights given - must be exactly 4')

  return damerauLevenshtein(input_string, 
                            target_string,
                            insertWeight=weights[0],
                            deleteWeight=weights[1],             
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
  algorithms = {'levenshtein': calc_levenshtein_distance,
                'damerau-levenshtein': calc_damerau_levenshtein_distance,
                'jaro': calc_jaro_distance,
                'jaro-winkler': calc_jaro_winkler_distance
               }

  return algorithms[algorithm](input_string, target_string, weights)

def find_best_match_for_string(input_string, target_string_list, function='levenshtein', weights=(1, 1, 1)):
  '''Find the best string match using Levenshtein distance'''
  
  if function in ['levenshtein', 'damerau-levenshtein', 'jaro', 'jaro-winkler']:
      best_estimate = ''
      best_distance = np.inf
      for target_string in target_string_list:
          distance = calc_distance(input_string, target_string, weights)
          
          if distance < best_distance:
              best_estimate = target_string
              best_distance = distance
              
  else:
      raise('Invalid input into function paramerter.')

  return best_estimate, best_distance
    
def find_best_match_for_list_of_strings(input_string_list: List[str],
                                        target_string_list: List[str],
                                        function: str = 'levenshtein',
                                        weights: Tuple(int) = (1, 1, 1)):
  '''Finds the accuracy score between a set of input strings and target strings'''
  
  matches = []
  for input_string in input_string_list:
      best_estimate, best_distance = find_best_match_for_string(input_string, target_string_list, function, weights)
      matches.append((input_string, best_estimate, best_distance))
      
  return matches

def calc_accuracy(y_pred: pd.Series, y_true: pd.Series):
  accuracy = (100 - len(set(y_true) - set(y_pred))) / 100
  return accuracy

def calc_accuracy_for_matches(input_string_list: List[str],
                              target_string_list: List[str],
                              y_true: pd.Series, 
                              weights: Tuple(int) = (1, 1, 1)):
  
  matches = find_best_match_for_list_of_strings(input_string_list, target_string_list, function, weights= weights)
  df_best_matches = pd.DataFrame(matches, columns=['Input', 'Estimate', 'Distance'])
  y_pred = df_best_matches['Estimate']
  accuracy = calc_accuracy(y_pred, y_true)

  return accuracy

def cross_val_levenshtein(input_string_list: List[str], 
                          target_string_list: List[str], 
                          y_true: pd.Series,
                          max_weights: Tuple(int), 
                          verbose: bool = True):

  weights_score = []
  for i in range(1, max_weights[0]+1):
          for j in range(1, max_weights[1]+1):
              for k in range(1, max_weights[2]+1):
                  matches = find_best_match_for_list_of_strings(input_string_list, target_string_list, function, weights=(i, j, k))
                  df_best_matches = pd.DataFrame(matches, columns=['Input', 'Estimate', 'Distance'])
                  y_pred = df_best_matches['Estimate']
                  accuracy = calc_accuracy(y_pred, y_true)
                  weights_score.append(((i, j, k), accuracy))

                  if verbose:
                      print('({}, {}, {}): {:.1%}'.format(i, j, k, accuracy))
  
  return weights_score

def cross_val_damerau_levenshtein(input_string_list: List[str], 
                                  target_string_list: List[str], 
                                  y_true: pd.Series,
                                  max_weights: Tuple(int), 
                                  verbose: bool = True):
  weights_score = []
  for i in range(1, max_weights[0]+1):
          for j in range(1, max_weights[1]+1):
              for k in range(1, max_weights[2]+1):
                  for m in range(1, max_weights[3]+1):
                      matches = find_best_match_for_list_of_strings(input_string_list, target_string_list, function, weights=(i, j, k, m))
                      df_best_matches = pd.DataFrame(matches, columns=['Input', 'Estimate', 'Distance'])
                      y_pred = df_best_matches['Estimate']
                      accuracy = calc_accuracy(y_pred, y_true)
                      weights_score.append(((i, j, k, m), accuracy))

                      if verbose:
                          print('({}, {}, {}, {}): {:.1%}'.format(i, j, k, m, accuracy))

  return weights_score

def cross_val_jaro(input_string_list: List[str], 
                    target_string_list: List[str], 
                    y_true: pd.Series,
                    max_weights: float = 0.0,
                    verbose: bool = True):
  weights_score = []
  matches = find_best_match_for_list_of_strings(input_string_list, target_string_list, function)
  df_best_matches = pd.DataFrame(matches, columns=['Input', 'Estimate', 'Distance'])
  y_pred = df_best_matches['Estimate']
  accuracy = calc_accuracy(y_pred, y_true)
  weights_score.append(accuracy)

  if verbose:
      print('{:.1%}'.format(accuracy))

  return weights_score

def cross_val_jaro_winkler(input_string_list: List[str], 
                          target_string_list: List[str], 
                          y_true: pd.Series,
                          weights: np.ndarray = np.linspace(0, 1, 10),
                          verbose: bool = True):
  weights_score = []
  for i in weights:
    matches = find_best_match_for_list_of_strings(input_string_list, target_string_list, function, weights=i)
    df_best_matches = pd.DataFrame(matches, columns=['Input', 'Estimate', 'Distance'])
    y_pred = df_best_matches['Estimate']
    accuracy = calc_accuracy(y_pred, y_true)
    weights_score.append((i, accuracy))

    if verbose:
        print('{}: {:.1%}'.format(i, accuracy))
  
  return weights_score

def cross_val_weights(input_string_list, target_string_list, y_true, algorithm='levenshtein', max_weights=(1, 1, 1), verbose=False):
  '''Cross-validation for weights in Levenshtein-distance algorithm'''
  
  if algorithm not in ['levenshtein', 'damerau-levenshtein', 'jaro', 'jaro-winkler']:
      raise Exception('Invalid function or weights provided.')

  algorithms = {'levenshtein': cross_val_levenshtein,
                'damerau-levenshtein': cross_val_damerau_levenshtein,
                'jaro': cross_val_jaro,
                'jaro-winkler': cross_val_jaro_winkler
               }
  
  weights_score = algorithms[algorithm](input_string_list, target_string_list, y_true, max_weights, verbose)
              
  return weights_score
