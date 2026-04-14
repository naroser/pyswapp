import sqlite3
import numpy as np
import pandas as pd
import math
import io

class StdevFunc:
    """SQLITE aggregate stdev."""
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None
        return math.sqrt(self.S / (self.k-1))

class SQL:
    """Handle an SQLite database."""
    def __init__(self, database):

        self.database = database

    # %% Connection
    def get_connection(self):
        """Create database connection."""
        con = sqlite3.connect(self.database)
        con.create_aggregate("STDEV", 1, StdevFunc)
        return con

    def to_sql(self, df, name, if_exists='fail', **kwargs):
        """Write data stored in a DataFrame to a SQL database."""

        with self.get_connection() as con:
            df.to_sql(name=name, con=con, if_exists=if_exists, **kwargs)

    def read_sql(self, sql):
        """Write SQL query or table into DataFrame."""

        with self.get_connection() as con:
            return pd.read_sql(sql,con=con)

    # %% Table management
    def _create_table(self, name, columns, types):
        """Create a SQL table"""

        if len(columns) != len(types):
            raise ValueError(
                f"Number of column names and types do not match: "
                f"{len(columns)} != {len(types)}")

        def is_valid_identifier(identifier):
            return identifier.replace("_", "").isalnum()

        if not is_valid_identifier(name):
            raise ValueError(f"Invalid table name: {name}")

        for col in columns:
            if not is_valid_identifier(col):
                raise ValueError(f"Invalid column name: {col}")

        # Build column definitions
        column_defs = ", ".join(f"{col} {typ}" for col, typ in zip(columns, types))
        sql = f"CREATE TABLE {name} ({column_defs});"

        # Execute SQL
        with self.get_connection() as con:
            con.execute(sql)

    def array_to_blob(self, arr):
        """Serialize ndarray to BLOB."""
        out = io.BytesIO()
        np.save(out, arr, allow_pickle=False)
        return out.getvalue()

    def blob_to_array(self, blob):
        """Deserialize SQLite BLOB to numpy array."""
        if isinstance(blob, (memoryview, bytearray)):
            blob = bytes(blob)
        out = io.BytesIO(blob)
        return np.load(out, allow_pickle=False)

    def _create_tables(self):
        """Initialize all database tables."""

        # amplitudes table
        self._create_table('amplitudes',
                           ['procset', 'wid', 'sin', 'rep', 'npts','dt','delay',
                            'sampling_rate','tapered_amps','rin_data', 'amps_data'],
                           ['TEXT', 'INT', 'INT', 'INT','INT',
                            'FLOAT','FLOAT','FLOAT','INT','BLOB', 'BLOB'])

        # dispersive_energy table
        self._create_table(
            'dispersive_energy',
            ['procset', 'wid', 'sin', 'rep', 'method',
             'velocity', 'wavenumber', 'f_id', 'f_value', 're', 'im'],
            ['TEXT', 'INT', 'INT', 'INT', 'TEXT',
             'FLOAT', 'FLOAT', 'INT', 'FLOAT', 'FLOAT', 'FLOAT']
        )

        # phase_differences table
        self._create_table('phase_differences',
                           ['procset', 'calc', 'sin', 'rep', 'wid', 'f_id', 'f_value', 'pd_data'],
                           ['TEXT', 'TEXT', 'INT', 'INT', 'INT', 'BLOB', 'BLOB', 'BLOB'])

        # curve table
        self._create_table('curves',
                           ['procset', 'wid', 'sin', 'rep', 'xmid',
                            'method', 'dc_mode', 'frequency', 'velocity', 'error'],
                           ['TEXT', 'INT', 'INT', 'INT', 'FLOAT',
                            'TEXT', 'INT', 'FLOAT', 'FLOAT', 'FLOAT'])

        # filter table
        self._create_table('filter',
                           ['procset', 'wid', 'sin', 'rep','key','x_value','y_value','type'],
                           ['TEXT', 'INT', 'INT', 'INT', 'TEXT', 'FLOAT', 'FLOAT', 'TEXT'])

    def get_table(self,name):
        """Return a table form the database as DataFrame."""
        return self.read_sql("""SELECT * FROM %s""" % name)

    def drop_table(self, name):
        """Drop table."""
        with self.get_connection() as con:
            con.execute(f"""DROP TABLE IF EXISTS {name}""")

    def show_tables(self):
        """Print all table names in the database."""
        tables = self.get_tables()
        print("Tables in project: " + ", ".join(tables))

    def check_data(self, table, params):
        """Check if entry exists already in database."""

        if table not in self.get_tables():
            return pd.DataFrame().empty

        sql = """SELECT procset, sin, rep
                  FROM %s
                  WHERE """ % table

        npar = len(params)
        for i, key in enumerate(params.keys()):

            if i < npar - 1:
                sql += f"{key}=={params[key]} AND "
            else:
                sql += f"{key}=={params[key]}"

        df = self.read_sql(sql)

        return df.empty

    def delete_data(self, table, params):
        """Delete entry from database."""

        with self.get_connection() as con:

            try:
                sql = """DELETE FROM %s WHERE """ % table
                npar = len(params)

                for i, key in enumerate(params.keys()):
                    if i < npar - 1:
                        sql += f"{key}=={params[key]} AND "
                    else:
                        sql += f"{key}=={params[key]}"
                con.execute(sql)

            except sqlite3.OperationalError:
                pass

    def get_tables(self):
        """Get all table names."""

        with self.get_connection() as con:
            table_names = con.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            table_list = [name[0] for name in table_names]
            return table_list

    # %% geometry information
    def read_geometry(self, geometry_file):
        """Read geometry CSV file and populate geometry, shots, and receivers tables."""

        # Read geometry file
        geom = pd.read_csv(geometry_file, delimiter=',', header=None,
                           names=['x', 'y', 'z', 'geophone', 'shots', 'first_geophone', 'num_geophones'])
        geom = geom.astype({'shots': str, 'first_geophone': str, 'num_geophones': str})
        geom.insert(0, 'station_id', np.arange(len(geom)) + 1)

        # Shots DataFrame
        shots = pd.DataFrame(geom.loc[geom.shots != '-1', ['station_id', 'shots']])
        shots['shots'] = shots['shots'].str.split(';')
        shots = shots.explode('shots')

        def _append_shots_df(shots_df, geom_df, col_name):
            tmp = pd.DataFrame(geom_df.loc[geom_df.shots != '-1', [col_name]])
            tmp[col_name] = tmp[col_name].str.split(';')
            tmp = tmp.explode(col_name).astype(int)
            shots_df[col_name] = tmp
            return shots_df

        shots = _append_shots_df(shots, geom, 'first_geophone')
        shots = _append_shots_df(shots, geom, 'num_geophones')

        shots.insert(2, 'rep', shots.groupby(['station_id', 'first_geophone', 'num_geophones']).cumcount() + 1)
        shots.insert(1, 'sin', (shots.rep.values == 1).cumsum())

        # Geophone indices (receivers)
        recs = pd.DataFrame(geom[geom.geophone > 0]['station_id'])
        recs.insert(0, 'rin', np.arange(len(recs)) + 1)
        recs = recs.astype({'rin': int, 'station_id': int})

        # Create tables
        self._create_table('geometry',
                           ['station_id', 'x', 'y', 'z', 'geophone',
                            'shots', 'first_geophone', 'num_geophones'],
                           ['INT', 'FLOAT', 'FLOAT', 'FLOAT', 'INT', 'INT', 'INT', 'INT'])
        self._create_table('shots', ['station_id', 'sin', 'rep',
                                     'shots', 'first_geophone', 'num_geophones'],
                           ['INT', 'INT', 'INT', 'INT', 'INT', 'INT'])
        self._create_table('receivers', ['rin', 'station_id'], ['INT', 'INT'])

        # write to database
        self.to_sql(geom, 'geometry', if_exists='replace',index=False)
        self.to_sql(shots, 'shots', if_exists='replace',index=False)
        self.to_sql(recs, 'receivers', if_exists='replace',index=False)

        # create all tables
        self._create_tables()

    def get_geometry(self, sin, rep=1):
        """Return source/receiver coordinates for one source location and shot index."""

        if sin != '*':
            sql = f"""SELECT s.sin, s.rep, s.first_geophone fg, s.num_geophones ng, 
                      g.x sx, g.y sy, g.z sz
                      FROM geometry g
                      INNER JOIN shots s ON s.station_id == g.station_id
                      WHERE s.rep=={rep} AND s.sin=={sin}"""
            sht = self.read_sql(sql)
            if len(sht) != 1:
                raise ValueError('Number of shots should be 1.')

            rec = pd.DataFrame(columns=['rin', 'rx', 'ry', 'rz'])
            for rin in range(sht.fg.item(), sht.fg.item() + sht.ng.item()):
                sql = f"""SELECT r.rin, g.x rx, g.y ry, g.z rz
                          FROM geometry g
                          INNER JOIN receivers r ON r.station_id == g.station_id
                          WHERE r.rin=={rin}"""
                rec = pd.concat([rec, self.read_sql(sql)])

        else:
            # Return all
            sql = """SELECT s.sin, s.rep, s.first_geophone fg, s.num_geophones ng, g.x sx, g.y sy, g.z sz
                     FROM geometry g
                     INNER JOIN shots s ON s.station_id == g.station_id"""
            sht = self.read_sql(sql)
            rec = pd.DataFrame(columns=['rin', 'rx', 'ry', 'rz'])
            for j in range(len(sht)):
                for rin in range(sht.fg.iloc[j], sht.fg.iloc[j] + sht.ng.iloc[j]):
                    sql = f"""SELECT r.rin, g.x rx, g.y ry, g.z rz
                              FROM geometry g
                              INNER JOIN receivers r ON r.station_id == g.station_id
                              WHERE r.rin=={rin}"""
                    rec = pd.concat([rec, self.read_sql(sql)])
            rec.drop_duplicates(inplace=True, ignore_index=True)

        return sht.reset_index(drop=True), rec.reset_index(drop=True)

    def get_shotfile(self, sin, rep=1):
        """Return the name of one shot file"""
        sql = f"""SELECT s.shots
                  FROM shots s
                  WHERE s.rep=={rep} AND s.sin=={sin}"""
        return self.read_sql(sql)

    # %% Interaction with settings
    def read_setting(self, settings):
        """Read settings and add to database"""
        self.to_sql(settings, 'settings', if_exists='replace', index = False)

    def get_settings(self):
        """Return the settings DataFrame"""
        return self.get_table('settings')

    # %% Interact with seismic data
    def get_proc_labels(self):
        sql = """SELECT DISTINCT procset
                 FROM amplitudes"""
        return self.read_sql(sql)['procset'].values

    def get_trafo_labels(self,procset, use_windows = False):

        if 'dispersive_energy' in self.get_tables():

            if use_windows:
                sql = """SELECT DISTINCT method
                         FROM dispersive_energy WHERE procset=='%s' AND wid != -1""" % procset
                labels = self.read_sql(sql)['method'].values
            else:
                sql = """SELECT DISTINCT method
                         FROM dispersive_energy WHERE procset=='%s' AND wid == -1""" % procset
                labels = self.read_sql(sql)['method'].values
            return labels

        return []

    def duplicate_data(self,data, sin, rep, wid=-1):

        params = {'sin': sin, 'rep': rep, 'procset': "'%s'" % 'tmp', 'wid': wid}
        if not self.check_data('amplitudes', params):
            self.delete_data('amplitudes', params)
        self.write_data(data, sin, rep, 'tmp', wid=wid)

    def write_data(self, data, sin, rep, procset, wid=-1):
        """Write amplitude data in TX domain to SQL table 'amplitudes'."""

        _, rec_geom = self.get_geometry(sin, rep)
        stream = data

        amps_data = self.array_to_blob(stream.st2amps().T)

        rin = pd.DataFrame({'rx': stream.receiver}).merge(rec_geom[['rx', 'rin']], on='rx', how='left')['rin'].values
        rin_data = self.array_to_blob(rin.astype(int))

        df = pd.DataFrame([{
            'procset': procset,
            'wid': wid,
            'sin': sin,
            'rep': rep,
            'npts': stream.npts,
            'dt': stream.dt,
            'delay': stream.delay,
            'sampling_rate': stream.sampling_rate,
            'tapered_amps': stream.tapered_amps,
            'rin_data': rin_data,
            'amps_data': amps_data
        }])

        # delete previous entry
        params = {'sin': sin, 'rep': rep, 'procset': f"'{procset}'", 'wid': wid}
        rows_exist = self.check_data('amplitudes', params)

        if not rows_exist and procset != 'raw':
            self.delete_data('amplitudes', params)

        self.to_sql(df, name='amplitudes', if_exists='append', index=False)

    def read_data(self, sin, rep, procset='proc1', wid=-1):
        """Read amplitude data from table 'amplitudes'."""

        sql = f"""
            SELECT *
            FROM amplitudes
            WHERE procset=='{procset}'
              AND wid=={wid}
              AND sin=={sin}
              AND rep=={rep}
        """
        df = self.read_sql(sql)

        if df.empty:
            return None, None, None, None

        rin = self.blob_to_array(df.iloc[0].rin_data)
        amps = self.blob_to_array(df.iloc[0].amps_data)

        rin_list = ",".join(map(str, rin.tolist()))

        sql_recs = f"""
            SELECT r.rin, g.x rx, g.y ry, g.z rz
            FROM geometry g
            INNER JOIN receivers r ON r.station_id = g.station_id
            WHERE r.rin IN ({rin_list})
            ORDER BY r.rin ASC
        """
        recs = self.read_sql(sql_recs)

        sql_sht = f"""
            SELECT s.sin, s.rep, s.first_geophone fg, s.num_geophones ng,
                   g.x sx, g.y sy, g.z sz
            FROM geometry g
            INNER JOIN shots s ON s.station_id = g.station_id
            WHERE s.rep=={rep} AND s.sin=={sin}
        """
        sht = self.read_sql(sql_sht)

        par = df[['npts', 'dt', 'delay', 'sampling_rate', 'tapered_amps']].iloc[[0]]

        return par, amps, recs, sht

    # %% FV data
    def write_FV(self, data, sin, rep, procset='proc1', wid=-1):
        """write dispersive energy data to SQL table 'dispersive_energy'"""

        cur_stream = data
        FV = cur_stream.dispersive_energy  # FV spectrum (vels, freq)
        freq = cur_stream.frequency  # frequency range
        cur_vel = cur_stream.velocity  # testing phase velocity
        cur_ks = cur_stream.wavenumber  # wavenumber
        method = cur_stream.trafo_type

        n_vel = len(cur_vel)
        n_freq = len(freq)

        rows = []
        for v in range(n_vel):
            for f in range(n_freq):
                rows.append({
                    'procset': procset,
                    'wid': wid,
                    'sin': sin,
                    'rep': rep,
                    'method': method,
                    'velocity': float(cur_vel[v]),
                    'wavenumber': float(cur_ks[v]),
                    'f_id': int(f),
                    'f_value': float(freq[f]),
                    're': float(FV[v, f].real),
                    'im': float(FV[v, f].imag),
                })

        df = pd.DataFrame(rows)

        params = {'sin': sin, 'rep': rep, 'procset': "'%s'" % procset, 'wid': wid, 'method': "'%s'" % method}
        if not self.check_data('dispersive_energy', params):
            self.delete_data('dispersive_energy', params)

        self.to_sql(df, name='dispersive_energy', if_exists='append', index=False)

    def read_FV(self, sin, rep, procset='proc1', method='phaseshift', wid=-1):
        """read dispersive energy data from SQL table 'dispersive_energy'"""

        if 'dispersive_energy' not in self.get_tables():
            return None, None, None, None

        sql = f"""
                    SELECT * FROM dispersive_energy
                    WHERE procset='{procset}'
                      AND wid={wid}
                      AND sin={sin}
                      AND rep={rep}
                      AND method='{method}'
                    ORDER BY velocity, f_id
                """

        df = self.read_sql(sql)
        if df.empty:
            return None, None, None, None

        velocities = df['velocity'].unique()
        freq_vals = df['f_value'].unique()

        n_vel = len(velocities)
        n_freq = len(freq_vals)

        # Reconstruct FV matrix
        FV = np.zeros((n_vel, n_freq), dtype=complex)

        for v_idx, vel in enumerate(velocities):
            block = df[df.velocity == vel]
            block = block.sort_values('f_id')
            try:
                FV[v_idx, :] = block['re'].values + 1j * block['im'].values
            except TypeError:
                FV[v_idx, :] = block['re'].values

        wavenumbers = df.groupby('velocity')['wavenumber'].first().values

        return velocities, wavenumbers, freq_vals, FV

    def wids_exist(self, procset):
        """check if windows exist"""

        sql = ("""SELECT DISTINCT wid
                 FROM amplitudes
                 WHERE procset=='%s'AND wid!=-1""" % procset)

        df = self.read_sql(sql)
        return True if not df.empty else False

    def get_wids(self, sin, rep, procset):
        """return the window ids for a sin/rep pair"""

        sql = ("""SELECT DISTINCT wid
                 FROM amplitudes
                 WHERE procset=='%s' AND sin==%d AND rep==%d AND wid!=-1""" % (procset, sin, rep))

        df = self.read_sql(sql)
        return sorted(df["wid"].tolist()) if not df.empty else []

    def write_pd(self, fids, freq, pd_data, sin, rep, procset='proc1'):
        """Write phase-difference data to SQL table 'phase_differences'."""

        row = pd.DataFrame([{
            "procset": procset,
            "calc": "NONE",
            "sin": sin,
            "rep": rep,
            "wid": -1,
            "f_id": self.array_to_blob(np.asarray(fids)),
            "f_value": self.array_to_blob(np.asarray(freq)),
            "pd_data": self.array_to_blob(np.asarray(pd_data))
        }])

        params = {
            "sin": sin,
            "rep": rep,
            "procset": f"'{procset}'",
            "wid": -1,
            "calc": "'NONE'"
        }

        if not self.check_data("phase_differences", params):
            self.delete_data("phase_differences", params)
        self.to_sql(row, name="phase_differences", if_exists="append", index=False)

    def group_pd(self, sin, procset='proc1', by='AVG'):
        """Group repeated measurements for a given sin. """

        sql = f"""
            SELECT procset, calc, sin, rep, wid, f_id, f_value, pd_data
            FROM phase_differences
            WHERE procset = '{procset}' AND sin = {sin} AND calc = 'NONE'
        """
        df = self.read_sql(sql)

        if df.empty:
            return pd.DataFrame()

        # Convert BLOBs → arrays
        df["fids"] = df["f_id"].apply(self.blob_to_array)
        df["freq"] = df["f_value"].apply(self.blob_to_array)
        df["pd"] = df["pd_data"].apply(self.blob_to_array)

        fids = df["fids"].iloc[0]
        freq = df["freq"].iloc[0]

        # Stack repeated pd arrays
        stacked_pd = np.stack(df["pd"].to_list(), axis=0)

        if by == "AVG":
            agg = np.nanmean(stacked_pd, axis=0)
        elif by == "STDEV":
            agg = np.nanstd(stacked_pd, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation: {by}")

        row = pd.DataFrame([{
            "procset": procset,
            "calc": by,
            "sin": sin,
            "rep": df["rep"].iloc[0],
            "wid": df["wid"].iloc[0],
            "f_id": self.array_to_blob(fids),
            "f_value": self.array_to_blob(freq),
            "pd_data": self.array_to_blob(agg)
        }])

        params = {
            "sin": sin,
            "procset": f"'{procset}'",
            "wid": -1,
            "calc": f"'{by}'"
        }

        if not self.check_data("phase_differences", params):
            self.delete_data("phase_differences", params)
        self.to_sql(row, name='phase_differences', if_exists="append", index=False)

        return row

    def read_pd(self, sin, procset='proc1', calc='NONE', columns = '*'):
        """Read phase-difference data from SQL table 'phase_differences'."""

        if not isinstance(columns, list):
            columns = list(columns)

        sql = (f"""SELECT {', '.join(columns)}
                 FROM phase_differences
                 WHERE procset=='%s'AND sin==%d AND calc=='%s'"""
               % (procset, sin, calc))

        blob_df = self.read_sql(sql)

        if blob_df.empty:
            return None

        return self.blob_to_array(blob_df.iloc[0].pd_data)

    def read_f_from_pd(self, procset='proc1', calc='NONE'):

        sql = f"""
            SELECT f_value
            FROM phase_differences
            WHERE procset='{procset}' AND calc='{calc}'
        """
        df = self.read_sql(sql)
        freq_arrays = [self.blob_to_array(b) for b in df["f_value"]]
        return sorted(set(np.concatenate(freq_arrays)))

    def write_curve(self, data, sin, rep, procset='proc1', wid=-1, xmid=0):
        """Write dispersion curve data to SQL table 'curves'."""

        df = pd.DataFrame({'procset':procset,
                            'wid': wid,
                            'sin': sin,
                            'rep': rep,
                            'xmid': data['xmid'],
                            'method': data['method'],
                            'dc_mode': data['dc_mode'],
                            'frequency': data['frequency'],
                            'velocity': data['velocity'],
                            'error': data['error']})

        params = {
            "procset": f"'{procset}'",
            "method": f"'{data['method']}'",
            "dc_mode": f"{data['dc_mode']}",
            "sin": sin,
            "rep": rep,
            "wid": wid,
            "xmid": xmid
        }

        if not self.check_data("curves", params):
            self.delete_data("curves", params)
        self.to_sql(df, name="curves", if_exists="append", index=False)

    def read_curve(self, params, cols = '*'):
        """Write dispersion curve data from SQL table 'curves'."""

        sql = f"SELECT {cols} FROM curves WHERE "
        sql += " AND ".join(f"{k}=={v}" for k, v in params.items())

        curve = self.read_sql(sql)
        return curve

    def read_curve_cc(self, params, cols = '*'):
        """Write dispersion curve data from SQL table 'curves'."""

        sql = f"SELECT {cols} FROM curves WHERE "
        sql += " AND ".join(f"{k}=={v}" for k, v in params.items())
        sql += " AND sin != -1 AND rep != -1"

        curve = self.read_sql(sql)
        return curve

    def write_filter(self, points, sin, rep, key = 't', procset = 'proc1', wid = -1, type = 'FK'):
        """Write filter to SQL table 'filter'."""

        x, y = zip(*sorted(points.items()))

        df = pd.DataFrame({'procset':procset,
                            'wid': wid,
                            'sin': sin,
                            'rep': rep,
                            'key': key,
                           'x_value':x,
                           'y_value':y,
                           'type':type})

        self.to_sql(df,name = 'filter', if_exists = 'append', index = False)

    def read_filter(self, params):
        """Read filter from SQL table 'filter'."""

        sql = "SELECT * FROM filter WHERE "
        sql += " AND ".join(f"{k}=={v}" for k, v in params.items())

        filter = self.read_sql(sql)

        if not filter.empty:
            filter.sort_values(['key', 'x_value'], ascending=[True, True],inplace=True)
            filter.reset_index(inplace = True, drop = True)

        return filter

