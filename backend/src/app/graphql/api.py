import strawberry
from strawberry.fastapi import GraphQLRouter
from typing import List

@strawberry.type
class User:
    id: int
    name: str

@strawberry.type
class Query:
    @strawberry.field
    async def users(self) -> List[User]:
        def get_users_from_mongodb():
            return {"test": "data"}
        return await get_users_from_mongodb()

schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)