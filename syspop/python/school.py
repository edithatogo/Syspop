
from logging import getLogger
from uuid import uuid4

from pandas import DataFrame

logger = getLogger()

def create_school(school_data: DataFrame, max_student_num: int = 30) -> DataFrame:
    """
    Transforms raw school data into a structured DataFrame of individual school entities.

    Each row in the input `school_data` is processed to create a dictionary
    representing a school. A unique ID is generated for each school, prefixed
    by its 'sector'. If a school's 'max_students' is less than `max_student_num`,
    it's set to `max_student_num`.

    Args:
        school_data (DataFrame): A DataFrame containing raw school information.
            Expected columns: 'area', 'age_min', 'age_max', 'sector',
            'latitude', 'longitude', 'max_students'.
        max_student_num (int, optional): The minimum value for 'max_students'.
            If a school's 'max_students' is below this, it's adjusted upwards.
            Defaults to 30.

    Returns:
        DataFrame: A DataFrame where each row is a school. Columns:
            'area_school' (renamed from 'area'), 'age_min', 'age_max',
            'latitude', 'longitude', 'max_students', 'school' (unique ID).
    """
    schools = []

    schools = []
    for _, row in school_data.iterrows():
        area = row["area"]
        age_min = row["age_min"]
        age_max = row["age_max"]
        sector = row["sector"]
        latitude = row["latitude"]
        longitude = row["longitude"]
        max_students = row["max_students"]

        if max_students < max_student_num:
            max_students = max_student_num

        name = f"{sector}_{str(uuid4())[:6]}"
        schools.append({
            "area_school": int(area),
            "age_min": int(age_min),
            "age_max": int(age_max),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "max_students": int(max_students),
            "school": str(name)
        })

    return DataFrame(schools)
