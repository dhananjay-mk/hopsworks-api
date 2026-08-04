#
#   Copyright 2026 Hopsworks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import json
import warnings
from io import BytesIO

import fastavro
import pandas as pd
import pytest
from hsfs import feature_group
from hsfs.core import kafka_engine
from hsfs.core.feature_group_engine import FeatureGroupEngine
from hsfs.engine import python


AVRO_SCHEMA = (
    '{"type":"record","name":"test_fg","namespace":"test_featurestore.db","fields":'
    '[{"name":"id","type":["null","long"]},'
    '{"name":"state","type":["null","string"]},'
    '{"name":"measurement","type":["null","double"]}]}'
)


class TestOnlineDeleteFillValues:
    def test_fills_non_primary_key_fields_with_null(self, mocker):
        fg = mocker.Mock()
        fg.primary_key = ["id"]
        fg.avro_schema = AVRO_SCHEMA

        assert kafka_engine._online_delete_fill_values(fg) == {
            "state": None,
            "measurement": None,
        }

    def test_composite_primary_key_excluded(self, mocker):
        fg = mocker.Mock()
        fg.primary_key = ["id", "state"]
        fg.avro_schema = AVRO_SCHEMA

        assert kafka_engine._online_delete_fill_values(fg) == {"measurement": None}


def _stream_online_fg(mocker, time_travel_format="DELTA"):
    mocker.patch("hopsworks_common.client._get_instance")
    fg = feature_group.FeatureGroup(
        name="test",
        version=1,
        featurestore_id=99,
        primary_key=["id"],
        partition_key=[],
        id=10,
        stream=True,
        online_enabled=True,
        time_travel_format=time_travel_format,
    )
    fg.primary_key = ["id"]
    return fg


class TestRemoveRowsStreamFeatureGroup:
    # ICEBERG alongside DELTA because _resolve_stream_python makes stream=True the default
    # for ICEBERG on external clients, so it is the common stream case rather than an edge one.
    @pytest.mark.parametrize("time_travel_format", ["DELTA", "ICEBERG"])
    def test_online_delete_runs_for_stream_fg(self, mocker, time_travel_format):
        mocker.patch("hsfs.engine._get_type", return_value="python")
        offline = mocker.patch.object(FeatureGroupEngine, "_commit_delete")
        online = mocker.patch.object(FeatureGroupEngine, "_delete_online_records")
        fg = _stream_online_fg(mocker, time_travel_format)

        fg.remove_rows(pd.DataFrame({"id": [2]}), delete_online=True)

        offline.assert_called_once()
        online.assert_called_once()

    def test_online_delete_by_default(self, mocker):
        # remove_rows deletes from both stores by default, mirroring insert.
        mocker.patch("hsfs.engine._get_type", return_value="python")
        offline = mocker.patch.object(FeatureGroupEngine, "_commit_delete")
        online = mocker.patch.object(FeatureGroupEngine, "_delete_online_records")
        fg = _stream_online_fg(mocker)

        fg.remove_rows(pd.DataFrame({"id": [2]}))

        offline.assert_called_once()
        online.assert_called_once()

    def test_offline_delete_only_when_delete_online_is_false(self, mocker):
        offline = mocker.patch.object(FeatureGroupEngine, "_commit_delete")
        online = mocker.patch.object(FeatureGroupEngine, "_delete_online_records")
        fg = _stream_online_fg(mocker)

        fg.remove_rows(pd.DataFrame({"id": [2]}), delete_online=False)

        offline.assert_called_once()
        online.assert_not_called()

    def test_deprecated_commit_delete_record_stays_offline_only(self, mocker):
        # The deprecated alias keeps the offline-only behaviour it had before the online
        # delete existed, so an existing caller does not start deleting online rows.
        offline = mocker.patch.object(FeatureGroupEngine, "_commit_delete")
        online = mocker.patch.object(FeatureGroupEngine, "_delete_online_records")
        fg = _stream_online_fg(mocker)

        fg.commit_delete_record(pd.DataFrame({"id": [2]}))

        offline.assert_called_once()
        online.assert_not_called()


class TestRemoveRowsValidation:
    def test_online_delete_on_non_online_fg_fails_before_offline_commit(self, mocker):
        from hopsworks_common.client.exceptions import FeatureStoreException

        mocker.patch("hopsworks_common.client._get_instance")
        commit = mocker.patch.object(FeatureGroupEngine, "_commit_delete")
        fg = feature_group.FeatureGroup(
            name="test",
            version=1,
            featurestore_id=99,
            primary_key=["id"],
            partition_key=[],
            id=10,
            stream=False,
            online_enabled=False,
            time_travel_format="DELTA",
        )
        fg.primary_key = ["id"]

        with pytest.raises(FeatureStoreException, match="not online-enabled"):
            fg.remove_rows(pd.DataFrame({"id": [2]}), delete_online=True)

        # offline store must not be mutated when the online-delete request is invalid
        commit.assert_not_called()


class TestCommitDeleteRecordDeprecated:
    def test_commit_delete_record_warns_and_delegates(self, mocker):
        mocker.patch.object(FeatureGroupEngine, "_commit_delete")
        online = mocker.patch.object(FeatureGroupEngine, "_delete_online_records")
        remove_rows = mocker.spy(feature_group.FeatureGroup, "remove_rows")
        fg = _stream_online_fg(mocker)
        delete_df = pd.DataFrame({"id": [2]})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fg.commit_delete_record(delete_df, delete_online=True)

        # the deprecation warning fired
        assert any(
            "commit_delete_record is deprecated" in str(w.message) for w in caught
        )
        # and it delegated to remove_rows, which still performed the delete
        remove_rows.assert_called_once()
        online.assert_called_once()


class TestDeleteDataframeKafka:
    def _make_feature_group(self, mocker):
        mocker.patch("hopsworks_common.client._get_instance")
        mocker.patch("hsfs.core.kafka_engine._get_kafka_config", return_value={})
        mocker.patch("hsfs.core.kafka_engine.Producer")
        # bypass the online-ingestion round trip in header setup; the delete leg
        # adds the operation header itself.
        mocker.patch("hsfs.core.kafka_engine._get_headers", return_value={})
        mocker.patch(
            "hsfs.feature_group.FeatureGroup.get_complex_features", return_value=[]
        )

        fg = feature_group.FeatureGroup(
            name="test",
            version=1,
            featurestore_id=99,
            primary_key=["id"],
            partition_key=[],
            id=10,
            stream=False,
        )
        # id=10 makes the constructor derive the key from (absent) features, so
        # set it explicitly to model a backend-initialized online feature group.
        fg.primary_key = ["id"]
        fg.feature_store = mocker.Mock()
        fg.feature_store.project_id = 234
        fg._subject = {"id": 1, "schema": AVRO_SCHEMA}
        fg._online_topic_name = "test_topic"
        return fg

    def test_primary_key_only_dataframe_serializes_with_null_fields(self, mocker):
        produced = {}

        def fake_produce(**kwargs):
            produced.update(kwargs)

        mocker.patch("hsfs.core.kafka_engine._kafka_produce", side_effect=fake_produce)
        fg = self._make_feature_group(mocker)

        python.Engine()._delete_dataframe_kafka(fg, pd.DataFrame({"id": [7]}), {})

        assert produced["key"] == "7"
        assert produced["headers"]["operation"] == b"delete"
        # No storage header. It gates the online leg alone ("0" makes OnlineFS skip the
        # row), so it cannot mark a record online-only, and absent already ingests.
        assert "storage" not in produced["headers"]

        with BytesIO(produced["encoded_row"]) as outf:
            record = fastavro.schemaless_reader(
                outf, fastavro.parse_schema(json.loads(AVRO_SCHEMA))
            )
        assert record == {"id": 7, "state": None, "measurement": None}

    def test_extra_non_key_columns_are_ignored(self, mocker):
        produced = {}
        mocker.patch(
            "hsfs.core.kafka_engine._kafka_produce",
            side_effect=lambda **kwargs: produced.update(kwargs),
        )
        fg = self._make_feature_group(mocker)

        python.Engine()._delete_dataframe_kafka(
            fg, pd.DataFrame({"id": [7], "state": ["nevada"]}), {}
        )

        # The online delete is by primary key, so a non-key column the caller passes
        # is ignored and serialized as null rather than overriding the tombstone fill.
        with BytesIO(produced["encoded_row"]) as outf:
            record = fastavro.schemaless_reader(
                outf, fastavro.parse_schema(json.loads(AVRO_SCHEMA))
            )
        assert record == {"id": 7, "state": None, "measurement": None}

    def test_entry_count_is_sent_by_default(self, mocker):
        fg = self._make_feature_group(mocker)
        mocker.patch("hsfs.core.kafka_engine._kafka_produce")
        get_headers = mocker.patch(
            "hsfs.core.kafka_engine._get_headers", return_value={}
        )

        python.Engine()._delete_dataframe_kafka(fg, pd.DataFrame({"id": [7, 8]}), {})

        assert get_headers.call_args[0][1] == 2

    def test_disable_online_ingestion_count_skips_the_entry_count(self, mocker):
        fg = self._make_feature_group(mocker)
        mocker.patch("hsfs.core.kafka_engine._kafka_produce")
        get_headers = mocker.patch(
            "hsfs.core.kafka_engine._get_headers", return_value={}
        )

        python.Engine()._delete_dataframe_kafka(
            fg,
            pd.DataFrame({"id": [7, 8]}),
            {"online_ingestion_options": {"disable_online_ingestion_count": True}},
        )

        assert get_headers.call_args[0][1] is None
