import re

import numpy as np
import pandas as pd
from sqlalchemy import Column, Integer, DateTime, Float, ForeignKey, String
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import relationship


class Base(object):
    @declared_attr
    def __tablename__(cls):
        name = cls.__name__
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    id = Column(Integer, primary_key=True)


Base = declarative_base(cls=Base)


class Result(Base):
    name = Column(String)

    start_time = Column(DateTime)
    end_time = Column(DateTime)

    accuracy = Column(Float)
    f_measure = Column(Float)
    adjusted_rand_score = Column(Float)
    silhouette = Column(Float)
    initial_n_features = Column(Integer)
    final_n_features = Column(Integer)

    args = relationship("Arg", back_populates="result")
    confusion_matrix = relationship("ConfusionMatrix", uselist=False, back_populates="result")
    selected_features = relationship("SelectedFeature", uselist=True)

    def details(self):
        args_str = '\n\t\t'.join(list(map(str, self.args)))
        features_str = '\n\t\t'.join(list(map(str, self.selected_features)))
        return f'''{self.id}: {self.name} - {self.start_time}
    F-Measure   {self.f_measure}
    Accuracy    {self.accuracy}
    Args:
        {args_str}
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


class Arg(Base):
    result_id = Column(Integer, ForeignKey('result.id'), nullable=False)
    name = Column(String)
    value = Column(String)

    result = relationship("Result", back_populates="args")

    def __str__(self):
        return f'{self.name}: {self.value}'


class ConfusionMatrix(Base):
    result_id = Column(Integer, ForeignKey('result.id'), nullable=False)

    result = relationship("Result", back_populates="confusion_matrix")
    confusion_matrix_labels = relationship("ConfusionMatrixLabel", back_populates="confusion_matrix")
    confusion_matrix_numbers = relationship("ConfusionMatrixNumber", back_populates="confusion_matrix")

    def __str__(self):
        labels = sorted(self.confusion_matrix_labels, key=lambda x: x.row_column)
        labels = [label.label for label in labels]

        matrix_dim = len(labels)
        matrix = np.zeros((matrix_dim, matrix_dim))
        for number in self.confusion_matrix_numbers:
            matrix[number.row, number.column] = number.value
        df = pd.DataFrame(matrix, columns=labels, index=labels)
        return df.to_string()


class ConfusionMatrixLabel(Base):
    confusion_matrix_id = Column(Integer, ForeignKey('confusion_matrix.id'), nullable=False)
    label = Column(String)
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
    column = Column(String)

    def __str__(self):
        return str(self.column)


class ClusterLabel(Base):
    result_id = Column(Integer, ForeignKey('result.id'), nullable=False)
    label = Column(String)
