import re
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy import Column, Integer, DateTime, Float, ForeignKey, String, PickleType, create_engine
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import relationship, sessionmaker


def get_conn_string(db_file):
    return f'sqlite:///{db_file}'


def local_create_engine(db_file):
    return create_engine(get_conn_string(db_file), echo=False)


def create_if_not_exists(db_file):
    engine = local_create_engine(db_file)
    Base.metadata.create_all(engine)


def local_create_session(db_file):
    engine = local_create_engine(db_file)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


class Base(object):
    @declared_attr
    def __tablename__(cls):
        name = cls.__name__
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    id = Column(Integer, primary_key=True)


Base = declarative_base(cls=Base)


class Arg(Base):
    result_id = Column(Integer, ForeignKey('result.id'), nullable=False)
    name = Column(String(64))
    value = Column(String(64))

    result = relationship("Result", back_populates="args")

    def __str__(self):
        return f'{self.name}: {self.value}'


class ConfusionMatrix(Base):
    result_id = Column(Integer, ForeignKey('result.id'), nullable=False)

    result = relationship("Result", back_populates="confusion_matrix")
    confusion_matrix_labels = relationship("ConfusionMatrixLabel", back_populates="confusion_matrix")
    confusion_matrix_numbers = relationship("ConfusionMatrixNumber", back_populates="confusion_matrix")

    def as_dataframe(self) -> pd.DataFrame:
        labels = sorted(self.confusion_matrix_labels, key=lambda x: x.row_column)
        labels = [label.label for label in labels]

        matrix_dim = len(labels)
        matrix = np.zeros((matrix_dim, matrix_dim))
        for number in self.confusion_matrix_numbers:
            matrix[number.row, number.column] = number.value
        df = pd.DataFrame(matrix, columns=labels, index=labels)

        return df

    def __str__(self):
        return self.as_dataframe().to_string()


class ConfusionMatrixLabel(Base):
    confusion_matrix_id = Column(Integer, ForeignKey('confusion_matrix.id'), nullable=False)
    label = Column(String(256))
    row_column = Column(Integer)

    confusion_matrix = relationship("ConfusionMatrix", back_populates="confusion_matrix_labels")


class ConfusionMatrixNumber(Base):
    confusion_matrix_id = Column(Integer, ForeignKey('confusion_matrix.id'), nullable=False)

    confusion_matrix = relationship("ConfusionMatrix", back_populates="confusion_matrix_numbers")

    value = Column(Float)
    column = Column(Integer)
    row = Column(Integer)


class SelectedFeature(Base):
    result_id = Column(Integer, ForeignKey('result.id'), nullable=False)
    column = Column(String(256))

    def __str__(self):
        return str(self.column)


class ClusterLabel(Base):
    result_id = Column(Integer, ForeignKey('result.id'), nullable=False)
    label = Column(String(32))

    def __str__(self):
        return str(self.label)


class Result(Base):
    name = Column(String(32))

    start_time = Column(DateTime)
    end_time = Column(DateTime)

    accuracy = Column(Float)
    f_measure = Column(Float)
    adjusted_rand_score = Column(Float)
    silhouette = Column(Float)
    initial_n_features = Column(Integer)
    final_n_features = Column(Integer)
    individual_evaluations: pd.DataFrame = Column(PickleType)

    args: List[Arg] = relationship("Arg", back_populates="result", uselist=True)
    confusion_matrix: ConfusionMatrix = relationship("ConfusionMatrix", uselist=False)
    selected_features: List[SelectedFeature] = relationship("SelectedFeature", uselist=True)
    result_labels: List[ClusterLabel] = relationship("ClusterLabel", uselist=True)

    def details(self):
        args_str = '\n\t\t'.join(list(map(str, self.args)))
        features_str = '\n\t\t'.join(list(map(str, self.selected_features)))
        labels_str = ','.join(list(map(str, self.result_labels)))
        return f'''{self.id}: {self.name} - {self.start_time}
    F-Measure   {self.f_measure}
    Accuracy    {self.accuracy}
    Args:
        {args_str}
    Result Labels: {labels_str}
    Confusion Matrix:
    {self.confusion_matrix}
    Selected Features:
    {features_str}
    '''

    def __str__(self):
        args_str = '\n\t\t'.join(list(map(str, self.args)))
        return f'''{self.id}: {self.name} - {self.start_time}
    F-Measure   {self.f_measure}
    Accuracy    {self.accuracy}
    Args:
        {args_str}
    '''
