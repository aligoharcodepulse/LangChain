from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name:str = 'Ali'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=4, default=0, description="A decimal value representing" \
    "cgpa of a student")

new_student = {'name':'Ali Gohar', 'age':20, 'email':'abc@gmail.com','cgpa':3.5}

student = Student(**new_student)
print(student)


student_dict = dict(student)
print(student_dict['email'])

student_json = student.model_dump_json()
print(student_dict['cgpa'])


