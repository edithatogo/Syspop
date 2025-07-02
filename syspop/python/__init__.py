# MAPING_DIARY_CFG_LLM_DIARY: Maps location names from LLM-generated diaries
# to standardized Syspop location types.
# Used in `create_diary_single_person_llm`.
# Format: { "standard_syspop_location": ["llm_location_alias1", "llm_location_alias2", ...] }
MAPING_DIARY_CFG_LLM_DIARY = {
    "company": ["office"],  # LLM's "office" is mapped to Syspop's "company"
    "household": ["home"],  # LLM's "home" is mapped to Syspop's "household"
}

# DIARY_CFG: Configuration for rule-based diary generation.
# Defines activities, their probabilities, time constraints, and modifiers.
# - Top-level keys: Person types (e.g., "worker", "student", "default").
# - "random_seeds": List of activities to choose from if no other specific activity
#                   is selected based on weight and time constraints.
# - Each activity (e.g., "household", "company"):
#   - "weight": Base probability/propensity for this activity.
#   - "time_ranges": List of (start_hour, end_hour) tuples when this activity can occur.
#   - "age_weight": Optional dict mapping age ranges (e.g., "0-5") to weight multipliers.
#   - "time_weight": Optional dict mapping hour ranges (e.g., "7-8") to weight multipliers.
#   - "max_occurrence": Optional integer limiting how many times this activity can occur in a day.
DIARY_CFG = {
    "worker": {
        "random_seeds": ["household", "travel", "company"],
        "household": {
            "weight": 0.75,  # every 1.0 people, how many will do such an acitivity
            "time_ranges": [(0, 8), (15, 24)],
            "age_weight": None,
            "time_weight": {"0-5": 2.0, "5-6": 1.5, "20-23": 1.5},
            "max_occurrence": None,
        },
        "travel": {
            "weight": 0.15,
            "time_ranges": [(7, 9), (16, 19)],
            "age_weight": None,
            "time_weight": None,
            "max_occurrence": 6,
        },
        "company": {
            "weight": 0.8,
            "time_ranges": [(7, 18)],
            "age_weight": None,
            "time_weight": {"7-8": 0.5, "17-18": 0.3},
            "max_occurrence": None,
        },
        "supermarket": {
            "weight": 0.01,
            "time_ranges": [(17, 20)],
            "age_weight": None,
            "time_weight": {"17-18": 1.5},
            "max_occurrence": 1,
        },
        "restaurant": {
            "weight": 0.001,
            "time_ranges": [(11, 13), (18, 20)],
            "age_weight": None,
            "time_weight": {"11-13": 2.0, "18-20": 3.0},
            "max_occurrence": 1,
        },
        # "pharmacy": {
        #    "weight": 0.000001,
        #    "time_ranges": [(11, 17)],
        #    "age_weight": None,
        #    "time_weight": None,
        # },
    },
    "student": {
        "random_seeds": ["household"],
        "household": {
            "weight": 0.75,
            "time_ranges": [(0, 8), (15, 24)],
            "age_weight": None,
            "time_weight": {"0-6": 2.0, "20-21": 1.25, "21-23": 1.5},
            "max_occurrence": None,
        },
        "school": {
            "weight": 0.85,
            "time_ranges": [(9, 15)],
            "age_weight": None,
            "time_weight": None,
            "max_occurrence": None,
        },
        "supermarket": {
            "weight": 0.01,
            "time_ranges": [(12, 13), (16, 18)],
            "age_weight": None,
            "time_weight": {"16-17": 1.5},
            "max_occurrence": 1,
        },
        "restaurant": {
            "weight": 0.001,
            "time_ranges": [(11, 13), (18, 20)],
            "age_weight": None,
            "time_weight": {"11-13": 2.0, "18-20": 3.0},
            "max_occurrence": 1,
        },
        # "pharmacy": {
        #    "weight": 0.000001,
        #    "time_ranges": [(11, 17)],
        #    "age_weight": None,
        #    "time_weight": None,
        # },
    },
    "default": {
        "random_seeds": ["household", "supermarket"],
        "household": {
            "weight": 0.75,
            "time_ranges": [(0, 24)],
            "age_weight": {"0-5": 1.2, "70-80": 1.2, "80-999": 1.5},
            "time_weight": {"0-5": 2.0, "5-6": 1.5, "20-21": 1.5, "21-23": 2.0},
            "max_occurrence": None,
        },
        "supermarket": {
            "weight": 0.01,
            "time_ranges": [(8, 20)],
            "age_weight": {"0-5": 0.3, "60-70": 0.75, "70-80": 0.15, "80-999": 0.001},
            "time_weight": {"9-15": 3.0, "17-18": 0.15},
            "max_occurrence": 1,
        },
        "restaurant": {
            "weight": 0.001,
            "time_ranges": [(11, 13), (18, 20)],
            "age_weight": {"0-5": 0.1, "60-70": 1.2, "70-80": 0.3, "80-999": 0.00001},
            "time_weight": {"11-13": 1.5, "17-20": 3.0},
            "max_occurrence": 1,
        },
        # "pharmacy": {
        #    "weight": 0.000001,
        #    "time_ranges": [(11, 17)],
        #    "age_weight": None,
        #    "time_weight": None,
        # },
    },
}

# SHARED_SPACE_NEAREST_DISTANCE_KM: Defines the maximum search radius (in km)
# for finding the nearest shared spaces of a particular type from a household's area.
# Used in `find_nearest_shared_space_from_household`.
# Format: { "shared_space_type": distance_in_km }
SHARED_SPACE_NEAREST_DISTANCE_KM = {
    "restaurant": 5.0,
    "fast_food": 5.0,
    "pub": 5.0,
    "supermarket": 10.0,
    "bakery": 5.0,
    "cafe": 5.0,
    "department_store": 15.0,
    "wholesale": 10.0,
    "park": 10.0,
    "hospital": 50.0,
}

# NZ_DATA_DEFAULT: Default directory path for New Zealand specific test data.
NZ_DATA_DEFAULT = "etc/data/test_data"
