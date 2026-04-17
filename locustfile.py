from locust import HttpUser, task, between
import random

LOAN_TYPES = ["Cash loans", "Revolving loans"]
EDUCATIONS = ["Higher education", "Secondary", "Incomplete higher", "Lower secondary"]
FAMILY = ["Married", "Single", "Separated", "Widow"]

class CreditRiskUser(HttpUser):
    wait_time = between(1, 2)

    def random_payload(self):
        age = random.randint(18, 70)
        emp = random.randint(0, min(age - 18, 30))
        return {
            "loan_type": random.choice(LOAN_TYPES),
            "age_years": age,
            "income_total": random.randint(20000, 500000),
            "loan_amount": random.randint(50000, 2000000),
            "employment_years": emp,
            "education": random.choice(EDUCATIONS),
            "family_status": random.choice(FAMILY),
            "owns_property": random.choice(["Y", "N"]),
            "owns_car": random.choice(["Y", "N"]),
            "ext_source_2": round(random.uniform(0.1, 0.9), 2)
        }

    @task(4)
    def predict_single(self):
        self.client.post("/predict", json=self.random_payload())

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def get_audit(self):
        self.client.get("/audit")

    @task(1)
    def get_stats(self):
        self.client.get("/audit/stats")

    @task(1)
    def invalid_age(self):
        payload = self.random_payload()
        payload["age_years"] = 150
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True
        ) as r:
            if r.status_code == 422:
                r.success()

    @task(1)
    def invalid_income(self):
        payload = self.random_payload()
        payload["income_total"] = -1000
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True
        ) as r:
            if r.status_code == 422:
                r.success()
