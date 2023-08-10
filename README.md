
# Code for smart grid set up and pure algorithm

'algo_5_20_5_5.py' is for pure algorithm setting with No. of stages = 5, No. of states = 20, No. of actions of player1 = 5, No. of actions of player2 = 5. Similarly for other files.

'smart_6_96_8_8.py' is for smart grid setting with No. of stages = 6, No. of states = 144, No. of actions of microgrid1 = 12, No. of actions of microgrid2 = 12. Similarly for other files.






## Dependencies and files details

'algo_error.py' should be run only after 'algo_5_20_5_5.py', 'algo_5_40_5_5.py' and 'algo_5_4_3_3.py'

'smartGrid_error.py' and 'equilibrium_deviation_grid.py' should be run only after 'smart_6_144_12_12.py', 'smart_6_36_6_6.py' and 'smart_6_96_8_8.py'

'smartGrid_error.py' gives the error plot for all the 3 three cases in a single figure.

'equilibrium_deviation_grid.py' gives the values when baseline policies are used for deviation. Three cases can be done by setting variable 'file_number' to 1,2,3.

'algo_error.py' gives the error plots for all the three cases in a single figure.

'lp_solve.py' is the game solver. Keep it in the folder.
## Acknowledgements
 
 https://github.com/rahulsavani/zerosum_matrix_game_solver
