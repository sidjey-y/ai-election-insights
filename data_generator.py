import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class DataGenerator:
    def __init__(self, start_date=None, end_date=None, seed=42):
        """Initialize the data generator with optional date range and random seed."""
        random.seed(seed)
        np.random.seed(seed)
        
        self.candidates = [
            "Alex Johnson", 
            "Taylor Smith", 
            "Jordan Williams", 
            "Morgan Brown"
        ]
        
        if start_date is None:
            self.start_date = datetime.now() - timedelta(days=30)
        else:
            self.start_date = start_date
            
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = end_date
        
        self.positive_templates = [
            "I really like {candidate}'s position on {issue}.",
            "{candidate} did a great job in the debate discussing {issue}.",
            "{candidate} has my vote because of their stance on {issue}.",
            "I'm impressed by how {candidate} handled questions about {issue}.",
            "{candidate} seems most qualified to address {issue}.",
            "Finally someone like {candidate} is talking about {issue}!",
            "I support {candidate} because they have concrete plans for {issue}.",
            "{candidate}'s leadership on {issue} is exactly what we need.",
        ]
        
        self.negative_templates = [
            "I don't agree with {candidate}'s position on {issue}.",
            "{candidate} completely missed the point about {issue} in the debate.",
            "{candidate} doesn't seem to understand {issue} at all.",
            "I'm disappointed by {candidate}'s stance on {issue}.",
            "{candidate} has no clear plan for dealing with {issue}.",
            "{candidate} is wrong about {issue} and it shows.",
            "I can't support {candidate} because of their views on {issue}.",
            "{candidate}'s approach to {issue} would be a disaster.",
        ]
        
        self.neutral_templates = [
            "I'm still undecided about {candidate}'s position on {issue}.",
            "{candidate} needs to clarify their stance on {issue}.",
            "I want to hear more from {candidate} about {issue}.",
            "Not sure if {candidate} has the right approach to {issue}.",
            "{candidate} mentioned {issue}, but didn't go into details.",
            "How will {candidate} actually implement their ideas on {issue}?",
            "Does {candidate} have experience dealing with {issue}?",
            "I'd like to compare {candidate}'s position on {issue} with the others.",
        ]
        
        self.issues = [
            "tuition fees",
            "campus housing",
            "student clubs funding",
            "mental health services",
            "sustainability initiatives",
            "diversity and inclusion",
            "campus safety",
            "academic resources",
            "internship opportunities",
            "student government transparency",
            "meal plan options",
            "transportation services",
            "campus recreation facilities",
            "technology resources",
            "student healthcare"
        ]
        

        self.candidate_bias = {
            "Alex Johnson": 0.6,     
            "Taylor Smith": 0.4,     
            "Jordan Williams": 0.55,  
            "Morgan Brown": 0.45      
        }
        
        self.issue_bias = {}
        for issue in self.issues:
            self.issue_bias[issue] = random.uniform(0.3, 0.7)
            
        self.sources = ["Survey", "Twitter", "Instagram", "Facebook", "Campus Forum", "Email Feedback"]
        
    def _random_date(self):
        """Generate a random date within the specified range."""
        delta = self.end_date - self.start_date
        random_days = random.randint(0, delta.days)
        return self.start_date + timedelta(days=random_days)
    
    def _generate_feedback(self, candidate, sentiment_type):
        """Generate a feedback comment for a given candidate and sentiment type."""
        issue = random.choice(self.issues)
        
        if sentiment_type == "positive":
            template = random.choice(self.positive_templates)
        elif sentiment_type == "negative":
            template = random.choice(self.negative_templates)
        else:
            template = random.choice(self.neutral_templates)
            
        return template.format(candidate=candidate, issue=issue)
    
    def _determine_sentiment(self, candidate, issue):
        """Determine if feedback should be positive, negative, or neutral based on biases."""
        candidate_factor = self.candidate_bias[candidate]
        issue_factor = self.issue_bias[issue]
        
        combined_factor = (candidate_factor + issue_factor) / 2 + random.uniform(-0.2, 0.2)
        
        if combined_factor > 0.6:
            return "positive"
        elif combined_factor < 0.4:
            return "negative"
        else:
            return "neutral"
    
    def generate_dataset(self, num_samples=500):
        """Generate a dataset of student feedback."""
        data = []
        
        for _ in range(num_samples):
            candidate = random.choice(self.candidates)
            issue = random.choice(self.issues)
            sentiment_type = self._determine_sentiment(candidate, issue)
            
            feedback = self._generate_feedback(candidate, sentiment_type)
            date = self._random_date()
            source = random.choice(self.sources)
            
            data.append({
                "date": date,
                "candidate": candidate,
                "feedback": feedback,
                "source": source
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("date")
        
        return df 