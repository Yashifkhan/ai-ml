# app/schemas.py
from pydantic import BaseModel

class DeveloperProfile(BaseModel):
    YearsCodePro: str
    EdLevel: str
    DevType: str
    OrgSize: str
    Industry: str
    RemoteWork: str
    Country: str
    LanguageHaveWorkedWith: str
    PlatformHaveWorkedWith: str
    DatabaseHaveWorkedWith: str
    ToolsTechHaveWorkedWith: str