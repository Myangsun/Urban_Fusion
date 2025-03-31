from app import UrbanFusionApp

app = UrbanFusionApp()
app.run_demo()

# result = app.process_query(
#     "Find optimal restaurant locations in Dubai with delivery radius of 3km, near residential areas, with low competition for Italian cuisine")
# print(result)

# result = app.analyze_data("path/to/your/data.csv")
# print(result)

# locations = [
#     {"lat": 25.2048, "lng": 55.2708, "rent": 5000,
#         "competition_count": 3, "score": 0.8},
#     {"lat": 25.2200, "lng": 55.3000, "rent": 7000,
#         "competition_count": 1, "score": 0.9}
# ]

# constraints = {
#     "max_rent": 6000,
#     "max_competition": 4
# }

# result = app.evaluate_locations(locations, constraints)
# print(result)
