from datetime import timedelta
from feast import BigQuerySource, Entity, FeatureView, Field
from feast.types import Float32

flower = Entity(name="flower_id", join_keys=["flower_id"])

iris_source = BigQuerySource(
    table="sound-memory-461003-g3.exam_demo.iris_dataset_01",
    timestamp_field="event_timestamp"
)

iris_feature = FeatureView(
    name="iris_feature",
    entities=[flower],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=iris_source
)
