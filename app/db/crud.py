from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import selectinload
from typing import List, Optional, Type, TypeVar, Generic
from datetime import datetime

from app.models.database import User, AITransaction, PredictionModel, APIKey

T = TypeVar("T")

class BaseCRUD(Generic[T]):
    def __init__(self, model: Type[T]):
        self.model = model

    async def get(self, db: AsyncSession, id: int) -> Optional[T]:
        result = await db.execute(select(self.model).where(self.model.id == id))
        return result.scalar_one_or_none()

    async def get_multi(self, db: AsyncSession, skip: int = 0, limit: int = 100) -> List[T]:
        result = await db.execute(select(self.model).offset(skip).limit(limit))
        return result.scalars().all()

    async def create(self, db: AsyncSession, obj_in: dict) -> T:
        db_obj = self.model(**obj_in)
        db.add(db_obj)
        await db.flush()
        await db.refresh(db_obj)
        return db_obj

    async def update(self, db: AsyncSession, id: int, obj_in: dict) -> Optional[T]:
        await db.execute(update(self.model).where(self.model.id == id).values(**obj_in))
        return await self.get(db, id)

    async def delete(self, db: AsyncSession, id: int) -> bool:
        result = await db.execute(delete(self.model).where(self.model.id == id))
        return result.rowcount > 0

    async def count(self, db: AsyncSession) -> int:
        result = await db.execute(select(func.count()).select_from(self.model))
        return result.scalar()

class UserCRUD(BaseCRUD[User]):
    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_by_username(self, db: AsyncSession, username: str) -> Optional[User]:
        result = await db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def get_active_users(self, db: AsyncSession, skip: int = 0, limit: int = 100) -> List[User]:
        result = await db.execute(
            select(User).where(User.is_active == True).offset(skip).limit(limit)
        )
        return result.scalars().all()

class TransactionCRUD(BaseCRUD[AITransaction]):
    async def get_by_user(self, db: AsyncSession, user_id: int, skip: int = 0, limit: int = 100) -> List[AITransaction]:
        result = await db.execute(
            select(AITransaction).where(AITransaction.user_id == user_id).offset(skip).limit(limit)
        )
        return result.scalars().all()

    async def get_by_model(self, db: AsyncSession, model_name: str) -> List[AITransaction]:
        result = await db.execute(
            select(AITransaction).where(AITransaction.model_name == model_name)
        )
        return result.scalars().all()

class ModelCRUD(BaseCRUD[PredictionModel]):
    async def get_by_name(self, db: AsyncSession, name: str) -> Optional[PredictionModel]:
        result = await db.execute(select(PredictionModel).where(PredictionModel.model_name == name))
        return result.scalar_one_or_none()

    async def get_deployed(self, db: AsyncSession) -> List[PredictionModel]:
        result = await db.execute(select(PredictionModel).where(PredictionModel.deployed == True))
        return result.scalars().all()

# Singleton instances
user_crud = UserCRUD(User)
transaction_crud = TransactionCRUD(AITransaction)
model_crud = ModelCRUD(PredictionModel)
