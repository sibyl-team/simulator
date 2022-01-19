import numpy as np

'''
Settings for town generation
'''

town_name = 'Parma' 

# Make sure to download country-specific population density data
# from https://data.humdata.org/organization/facebook
population_path='lib/data/population/population_ita_2019-07-01.csv' # Population density file

sites_path='lib/data/queries/' # Directory containing OSM site query details
bbox = (44.7691, 44.8229, 10.2740, 10.3811) # Coordinate bounding box # 

# Population per age group in the region (matching the RKI age groups)
# Source for Germany: https://www.citypopulation.de/en/germany/
population_per_age_group = np.array([
    16727,  # 0-9
    17660,  # 10-19
    20387,  # 20-29
    26066,  # 30-39
    30688,  # 40-49
    31469,  # 50-59
    22417,  # 60-69
    19074,  # 70-79
    12675,  # 80-89
    3055], # 90+
    dtype=np.int32) 

town_population = 200218
region_population = population_per_age_group.sum()

# Roughly 100k tests per day in Germany: https://www.rki.de/DE/Content/Infekt/EpidBull/Archiv/2020/Ausgaben/15_20.pdf?__blob=publicationFile
daily_tests_unscaled = int(170000 * (town_population / 60000000)) # Italy on the 6th of July 21

# Information about household structure (set to None if not available)
# Source for Germany: https://www.destatis.de/EN/Themes/Society-Environment/Population/Households-Families/Tables/lrbev05.html 
#household_info = {
#    'size_dist' : [41.9, 33.8, 11.9, 9.1, 3.4], # distribution of household sizes (1-5 people)
#    'soc_role' : {
#    'children' : [1, 1, 3/20, 0, 0, 0], # age groups 0,1,2 can be children 
#    'parents' : [0, 0, 17/20, 1, 0, 0], # age groups 2,3 can be parents
#    'elderly' : [0, 0, 0, 0, 1, 1] # age groups 4,5 are elderly
#    }
#}
household_info = None

