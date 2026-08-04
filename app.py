import re
import numpy as np
import joblib
import streamlit as st
from scipy.sparse import hstack

# =---------------------------------------------------------------------------=
# Labeler - used ONLY to explain the input (interpretability).
# =---------------------------------------------------------------------------=
DESCRIPTION_LF = {
    'LEAK':        ['LEAK','LEAKING','BOCOR','KEBOCORAN','KEBCORAN','REMBES','REMBESAN','DRIP'],
    'OVERHEAT':    ['OVERHEAT','OVER HEAT','OVERHT','OVERHEAD','HOT','PANAS','TEMP HIGH','TEMPERATURE','WRAM'],
    'LOW POWER':   ['LOW POWER','LOWPOWER','LOW PWR','POWERLESS','LEMAH','WEAK','KURANG TENAGA'],
    'CONSUMPTION': ['TOP UP','EMPTY','KOSONG','HABIS','LOW LEVEL','LOW OIL','LOW HYD','HYD LOW',
                    'LOW FUEL','OIL LOW','OLI LOW','LOW PRESSURE','FSS LOW'],
    'NO_START':    ['CANT START',"CAN'T START",'TIDAK BISA START','NO START','CANT',
                    'TIDAK BISA','SHUTDOWN','SHUT DOWN','OFF','MATI','PARKIR','STOP'],
    'BROKEN':      ['BROKEN','PUTUS','RUSAK','PATAH','CRACK','RETAK','BEND','BENGKOK',
                    'PECAH','MISSING','HILANG','FAIL','FAILED','MALFUNCTION','TUMPUL'],
    'LOOSE':       ['LOOSE','LOSE','KENDOR','LEPAS','COPOT','LONGGAR'],
    'STUCK':       ['STUCK','JAMMED','MACET','JAM','FROZEN','NGELOCK','LOCKED','LOCK',
                    'TIDAK GERAK','TIDAK BERFUNGSI','TIDAK MAU'],
    'ABNORMAL':    ['ABNORMAL','ABN','ERROR','EROR','WARNING','ALARM','FAULT','CODE',
                    'CONTAMINATION','CONTAMINASI','PRESSURE'],
    'NOISE':       ['NOISE','BUNYI','SUARA','BERISIK','SMOKE','TICK','KNOCK'],
}
ACTIVITY_LF = {
    'REPLACE':  ['REPLACE','GANTI','GNT'],
    'REPAIR':   ['REPAIR','RIPAIR','PERBAIKI','WELDING','WELD','REORING','ORING','REPOSISI','MODIF'],
    'INSPECT':  ['CHECK','CEK','CHEK','T/S','TS','INSPEKSI','INSPECTION','TROUBLESHOOT','TEST'],
    'ADJUST':   ['ADJUST','LEVELING','CALIBRATE','KALIBRASI','ALIGN','RETORQUE','STANDARISASI','TORQUE'],
    'INSTALL':  ['INSTALL','NSTALL','PASANG','MOUNT','CONNECT','CONECT','COUPLE'],
    'REMOVE':   ['REMOVE','LEPAS','CABUT','DISCONNECT','UNCOUPLE'],
    'CLEAN':    ['CLEANING','CLEAN','WASHING','WASH','BERSIHKAN','GREASING'],
    'REFILL':   ['ADD','TOP','RECHARGE','FILL','ISI','TAMBAH','RECOVERY'],
    'RESET':    ['RESET','RESTART','REBOOT'],
}

# Plain-language gloss for each labeler output, so the explanation is readable
# by a technician rather than being a bare code. [Rule 8]
SYMPTOM_GLOSS = {
    'LEAK':        {'id': 'Kebocoran',            'en': 'Leak'},
    'OVERHEAT':    {'id': 'Panas berlebih',       'en': 'Overheating'},
    'LOW POWER':   {'id': 'Kurang power',         'en': 'Low power'},
    'CONSUMPTION': {'id': 'Level/isi berkurang',  'en': 'Low level / consumption'},
    'NO_START':    {'id': 'Tidak bisa start/mati','en': 'Will not start / shut down'},
    'BROKEN':      {'id': 'Rusak/patah',          'en': 'Broken'},
    'LOOSE':       {'id': 'Kendor/lepas',         'en': 'Loose'},
    'STUCK':       {'id': 'Tersangkut',           'en': 'Stuck'},
    'ABNORMAL':    {'id': 'Tidak normal/error',   'en': 'Abnormal / error'},
    'NOISE':       {'id': 'Bunyi tidak normal',   'en': 'Unusual noise'},
    'UNLABELED':   {'id': 'Tidak dikenali',       'en': 'Not recognised'},
}
ACTION_GLOSS = {
    'REPLACE':   {'id': 'Ganti',        'en': 'Replace'},
    'REPAIR':    {'id': 'Perbaiki',     'en': 'Repair'},
    'INSPECT':   {'id': 'Periksa/cek',  'en': 'Inspect'},
    'ADJUST':    {'id': 'Setel',        'en': 'Adjust'},
    'INSTALL':   {'id': 'Pasang',       'en': 'Install'},
    'REMOVE':    {'id': 'Lepas',        'en': 'Remove'},
    'CLEAN':     {'id': 'Bersihkan',    'en': 'Clean'},
    'REFILL':    {'id': 'Isi/tambah',   'en': 'Refill'},
    'RESET':     {'id': 'Reset',        'en': 'Reset'},
    'UNLABELED': {'id': 'Tidak dikenali','en': 'Not recognised'},
}


def apply_lf(text, lfs):
    """Return the first label whose keyword matches, or UNLABELED if none do."""
    if text is None or str(text).strip() == '':
        return 'UNLABELED'
    s = str(text).upper()
    for label, kws in lfs.items():
        for kw in kws:
            if len(kw) <= 3:
                if re.search(r'\b' + re.escape(kw) + r'\b', s):
                    return label
            elif kw in s:
                return label
    return 'UNLABELED'


# =---------------------------------------------------------------------------=
# Interface strings. Bahasa Indonesia is the default; English is available via
# the language selector. Wording is kept parallel between the two. [Rule 1]
# =---------------------------------------------------------------------------=
T = {
    'id': {
        'title': 'Klasifikasi Penyebab Kerusakan Alat',
        'subtitle': 'Masukkan log kerusakan, aplikasi akan memprediksi Penyebab (Cause) '
                    'dan menjelaskan isi teks yang Anda masukkan.',
        'lang_label': 'Bahasa',
        'desc_label': 'Deskripsi kerusakan (gejala)',
        'desc_help': 'Tulis gejala yang dilaporkan, contoh: TRACK LH LOOSE',
        'act_label': 'Aktivitas perbaikan (tindakan)',
        'act_help': 'Tulis tindakan yang dilakukan, contoh: INSTALL & ADJUST TRACK',
        'obj_label': 'Komponen (Object)',
        'obj_help': 'Pilih sistem komponen yang mengalami kerusakan',
        'desc_helper': 'Tuliskan gejala kerusakan secara singkat, seperti yang dicatat teknisi. '
                       'Contoh: "AC PANAS", "TRACK LH LOOSE", "OVERHEAT".',
        'act_helper': 'Tuliskan tindakan perbaikan yang dilakukan. '
                      'Contoh: "REPLACE HOSE", "CLEAN UP FILTER AC", "CHECK WIRING".',
        'obj_helper': 'Sistem/komponen tempat kerusakan terjadi (mesin, AC, undercarriage, dll).',
        'example_label': 'Belum tahu harus menulis apa? Coba salah satu contoh:',
        'example_none': 'Pilih contoh...',
        'glossary_header': 'Panduan istilah alat berat (untuk orang awam)',
        'glossary_intro': 'Log kerusakan ditulis oleh teknisi dengan istilah teknis, campuran '
                          'Bahasa Indonesia dan Inggris. Berikut arti istilah yang sering muncul:',
        'submit': 'Prediksi Penyebab',
        'clear': 'Hapus Semua',
        'hint_enter': 'Tekan Enter pada kolom teks untuk langsung memprediksi.',
        'err_both_empty': 'Mohon isi Deskripsi dan/atau Aktivitas terlebih dahulu.',
        'warn_desc_empty': 'Kolom Deskripsi kosong. Prediksi hanya memakai Aktivitas, '
                           'sehingga hasilnya bisa kurang akurat.',
        'warn_act_empty': 'Kolom Aktivitas kosong. Prediksi hanya memakai Deskripsi, '
                          'sehingga hasilnya bisa kurang akurat.',
        'warn_too_short': 'Teks yang dimasukkan sangat pendek. Tambahkan keterangan '
                          'agar prediksi lebih dapat diandalkan.',
        'spinner': 'Memproses log...',
        'result_header': 'Hasil Prediksi',
        'predicted': 'Penyebab (Cause)',
        'your_input': 'Log yang Anda masukkan',
        'top3': 'Kandidat teratas',
        'other_note': '"OTHER" berarti penyebabnya di luar 10 kategori umum. '
                      'Mohon ditinjau secara manual.',
        'low_conf': 'Model kurang yakin pada log ini. Sebaiknya ditinjau manual.',
        'explain_header': 'Pembacaan teks Anda',
        'explain_note': 'Bagian ini hanya menjelaskan isi teks; prediksi di atas '
                        'tetap dihitung dari teks lengkap.',
        'symptom': 'Gejala pada Deskripsi',
        'action': 'Tindakan pada Aktivitas',
        'unrecognised': 'Sebagian teks tidak dikenali oleh penjelas. Prediksi tetap '
                        'menggunakan teks lengkap.',
        'done': 'Selesai. Ubah input di atas lalu tekan "Prediksi Penyebab" untuk log berikutnya, '
                'atau tekan "Hapus Semua" untuk mulai dari awal.',
        'history': 'Riwayat prediksi (sesi ini)',
        'history_empty': 'Belum ada prediksi pada sesi ini.',
        'causes_header': 'Daftar kategori Penyebab yang dikenali',
        'causes_note': 'Model mengenali 10 kategori berikut, ditambah OTHER untuk penyebab '
                       'yang jarang terjadi.',
        'model_error': 'File model "cause_classifier.joblib" tidak ditemukan atau tidak dapat '
                       'dibaca. Pastikan file berada di folder yang sama dengan app.py, '
                       'lalu muat ulang halaman.',
        'model_error_detail': 'Detail teknis',
    },
    'en': {
        'title': 'Machine Failure Cause Classifier',
        'subtitle': 'Enter a failure log; the app predicts the Cause and explains '
                    'what your text says.',
        'lang_label': 'Language',
        'desc_label': 'Failure description (symptom)',
        'desc_help': 'Describe the reported symptom, e.g. TRACK LH LOOSE',
        'act_label': 'Repair activity (action)',
        'act_help': 'Describe the action taken, e.g. INSTALL & ADJUST TRACK',
        'obj_label': 'Component (Object)',
        'obj_help': 'Select the component system involved',
        'desc_helper': 'Write the failure symptom briefly, the way a technician logs it. '
                       'Example: "AC PANAS", "TRACK LH LOOSE", "OVERHEAT".',
        'act_helper': 'Write the repair action taken. '
                      'Example: "REPLACE HOSE", "CLEAN UP FILTER AC", "CHECK WIRING".',
        'obj_helper': 'The system/component where the failure occurred (engine, AC, undercarriage, etc.).',
        'example_label': 'Not sure what to write? Try one of these examples:',
        'example_none': 'Choose an example...',
        'glossary_header': 'Heavy-equipment term guide (for non-specialists)',
        'glossary_intro': 'Failure logs are written by technicians using technical terms, mixing '
                          'Indonesian and English. Here are commonly seen terms:',
        'submit': 'Predict Cause',
        'clear': 'Clear',
        'hint_enter': 'Press Enter in a text field to predict immediately.',
        'err_both_empty': 'Please enter a Description and/or an Activity first.',
        'warn_desc_empty': 'The Description field is empty. The prediction uses only the '
                           'Activity, so it may be less reliable.',
        'warn_act_empty': 'The Activity field is empty. The prediction uses only the '
                          'Description, so it may be less reliable.',
        'warn_too_short': 'The text entered is very short. Add more detail for a more '
                          'reliable prediction.',
        'spinner': 'Processing the log...',
        'result_header': 'Prediction Result',
        'predicted': 'Cause',
        'your_input': 'The log you entered',
        'top3': 'Top candidates',
        'other_note': '"OTHER" means the cause falls outside the 10 common categories. '
                      'Please review it manually.',
        'low_conf': 'The model is less certain about this record. Consider reviewing it manually.',
        'explain_header': 'How your text was read',
        'explain_note': 'This section only explains the text; the prediction above is still '
                        'computed from the full text.',
        'symptom': 'Symptom in the Description',
        'action': 'Action in the Activity',
        'unrecognised': 'Part of the text was not recognised by the explainer. The prediction '
                        'still uses the full text.',
        'done': 'Done. Edit the fields above and press "Predict Cause" for the next log, '
                'or press "Clear" to start over.',
        'history': 'Prediction history (this session)',
        'history_empty': 'No predictions yet in this session.',
        'causes_header': 'Cause categories the model recognises',
        'causes_note': 'The model recognises the following 10 categories, plus OTHER for '
                       'rarely occurring causes.',
        'model_error': 'The model file "cause_classifier.joblib" could not be found or read. '
                       'Make sure it is in the same folder as app.py, then reload the page.',
        'model_error_detail': 'Technical details',
    },
}

# Preset example logs (real, in-distribution cases) so a general-public respondent
# can try the app without knowing heavy-equipment shorthand. [Rule 2 / onboarding]
EXAMPLE_LOGS = [
    ('Kebocoran selang hidrolik / Hydraulic hose leak',
     'HOSE TREVAL KANAN LEAK', 'REPLACE HOSE TRAVEL', 'HYD SYSTEMS'),
    ('Tenaga mesin lemah / Low engine power',
     'LOW POWER ENGINE', 'REPLACE FUEL FILTER', 'ENGINE'),
    ('AC panas / AC not cooling',
     'AC PANAS', 'CLEAN UP FILTER AC', 'AC SYSTEMS'),
    ('Track kendor / Loose track',
     'TRACK LH LOOSE', 'INSTALL & ADJUST TRACK LH', 'UNDERCARRIAGE'),
    ('Mesin overheat / Engine overheating',
     'OVERHEAT', 'WASHING RADIATOR, COOLER HYD', 'ENGINE'),
    ('Tidak bisa start / Will not start',
     'CANT START', 'CHECK WIRING STARTING', 'ELECTRICS'),
]

# Plain-language glossary of the shorthand that appears in the logs, bilingual.
GLOSSARY = [
    ('LEAK / BOCOR', 'Kebocoran cairan (oli, hidrolik) — a fluid leak'),
    ('OVERHEAT / PANAS', 'Suhu terlalu tinggi — component running too hot'),
    ('LOW POWER / LEMAH', 'Tenaga berkurang — reduced engine/hydraulic power'),
    ('TRACK', 'Rantai roda pada excavator/dozer — the crawler track'),
    ('UNDERCARRIAGE', 'Bagian bawah alat (track, roller, idler) — the running gear'),
    ('HOSE / SELANG', 'Selang saluran cairan — a fluid hose'),
    ('HYD / HYDRAULIC', 'Sistem hidrolik — the hydraulic system'),
    ('AC PANAS', 'AC tidak dingin / kepanasan — air-conditioning not cooling'),
    ('CANT START / MATI', 'Alat tidak bisa dinyalakan — the machine will not start'),
    ('LOOSE / KENDOR', 'Komponen longgar — a loose part'),
    ('REPLACE / GANTI', 'Mengganti komponen — replacing a part'),
    ('CHECK / CEK', 'Memeriksa — inspecting/checking'),
    ('ADJUST / SETEL', 'Menyetel ulang — adjusting'),
    ('CLEAN / BERSIHKAN', 'Membersihkan — cleaning'),
]

st.set_page_config(page_title='Machine Failure Log Classifier', page_icon='🔧',
                   layout='wide', initial_sidebar_state='collapsed')

# ---------------------------------------------------------------------------
# Session state: results persist across reruns so that changing a widget does
# not make the previous result disappear. [Rules 3, 4, 6]
# ---------------------------------------------------------------------------
if 'result' not in st.session_state:
    st.session_state.result = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Center all content in a wide-but-bounded middle column so the wide layout
# does not stretch the form edge to edge.
_pad_l, body, _pad_r = st.columns([1, 4, 1])

# Language selector - compact segmented control at the top right of the content,
# replacing the old sidebar panel. The user controls the interface language. [Rule 7]
with body:
    _spacer, _lang = st.columns([4, 1])
    with _lang:
        lang_choice = st.segmented_control(
            'Bahasa / Language',
            options=['id', 'en'],
            format_func=lambda c: 'ID' if c == 'id' else 'EN',
            default='id',
            label_visibility='collapsed',
        ) or 'id'
t = T[lang_choice]


# ---------------------------------------------------------------------------
# Load trained artifacts, with a readable message if the file is missing or was
# produced by a different scikit-learn version. [Rule 5]
# ---------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    return joblib.load('cause_classifier.joblib')


try:
    bundle = load_bundle()
except Exception as exc:  # missing file, version mismatch, corrupt pickle
    body.error(t['model_error'])
    with body.expander(t['model_error_detail']):
        st.code(f'{type(exc).__name__}: {exc}')
    st.stop()

tfidf          = bundle['tfidf']
obj_encoder    = bundle['obj_encoder']
classifier     = bundle['classifier']
label_encoder  = bundle['label_encoder']
object_options = bundle['object_options']

body.title(t['title'])
body.caption(t['subtitle'])


def clear_form():
    """Reset every field and the current result. [Rule 6]"""
    st.session_state.desc_input = ''
    st.session_state.act_input = ''
    st.session_state.obj_input = object_options[0]
    st.session_state.example_pick = 0
    st.session_state.result = None


def fill_example():
    """Populate the fields from the chosen preset example. Helps a general-public
    respondent who does not know heavy-equipment shorthand get valid input."""
    idx = st.session_state.get('example_pick')
    if idx is None or idx == 0:
        return
    _label, desc, act, ex_obj = EXAMPLE_LOGS[idx - 1]
    st.session_state.desc_input = desc
    st.session_state.act_input = act
    if ex_obj in object_options:
        st.session_state.obj_input = ex_obj
    st.session_state.result = None


# Example picker + glossary sit OUTSIDE the form, so selecting an example fills
# the fields immediately (a form would defer the callback until submit).
with body.container():
    st.selectbox(
        t['example_label'],
        options=list(range(len(EXAMPLE_LOGS) + 1)),
        format_func=lambda i: t['example_none'] if i == 0 else EXAMPLE_LOGS[i - 1][0],
        key='example_pick',
        on_change=fill_example,
    )
    with st.expander(t['glossary_header']):
        st.caption(t['glossary_intro'])
        for term, meaning in GLOSSARY:
            st.markdown(f"- **{term}** — {meaning}")

# ---------------------------------------------------------------------------
# Input form. Using st.form means Enter submits the form directly, which gives
# frequent users a keyboard shortcut instead of forcing a mouse click. [Rule 2]
# ---------------------------------------------------------------------------
with body.form('log_form'):
    description = st.text_input(t['desc_label'], key='desc_input',
                                placeholder='TRACK LH LOOSE', help=t['desc_help'])
    st.caption(t['desc_helper'])
    activity = st.text_input(t['act_label'], key='act_input',
                             placeholder='INSTALL & ADJUST TRACK', help=t['act_help'])
    st.caption(t['act_helper'])
    obj = st.selectbox(t['obj_label'], object_options, key='obj_input', help=t['obj_help'])
    st.caption(t['obj_helper'])
    col_a, col_gap, col_b = st.columns([2, 3, 2])
    with col_a:
        submitted = st.form_submit_button(t['submit'], type='primary', use_container_width=True)
    with col_b:
        st.form_submit_button(t['clear'], on_click=clear_form, use_container_width=True)
body.caption(t['hint_enter'])

# Reference list of recognised causes, always available without leaving the
# screen, so the user need not memorise the label space. [Rule 8]
with body.expander(t['causes_header']):
    st.caption(t['causes_note'])
    st.write(', '.join([c for c in label_encoder.classes_ if c != 'OTHER']))

# ---------------------------------------------------------------------------
# Handle a submission
# ---------------------------------------------------------------------------
if submitted:
    desc_text = (description or '').strip()
    act_text = (activity or '').strip()

    if not desc_text and not act_text:
        # Blocking error: nothing to classify. [Rule 5]
        body.error(t['err_both_empty'])
        st.session_state.result = None
    else:
        notices = []
        if not desc_text:
            notices.append(t['warn_desc_empty'])
        if not act_text:
            notices.append(t['warn_act_empty'])
        if len((desc_text + ' ' + act_text).strip()) < 5:
            notices.append(t['warn_too_short'])

        text = (desc_text + ' ' + act_text).strip()
        with st.spinner(t['spinner']):  # progress feedback [Rule 3]
            X = hstack([tfidf.transform([text]),
                        obj_encoder.transform([[obj]])]).tocsr()
            probs = np.ravel(classifier.predict_proba(X))
            order = np.argsort(probs)[::-1][:3]
            top3 = [(label_encoder.classes_[i], float(probs[i])) for i in order]

        st.session_state.result = {
            'cause': top3[0][0],
            'top3': top3,
            'desc': desc_text,
            'act': act_text,
            'obj': obj,
            'symptom': apply_lf(desc_text, DESCRIPTION_LF),
            'action': apply_lf(act_text, ACTIVITY_LF),
            'notices': notices,
        }
        # Keep the last five predictions so a frequent user can look back
        # without re-entering anything. [Rules 2, 8]
        st.session_state.history.insert(0, {
            'desc': desc_text, 'act': act_text,
            'obj': obj, 'cause': top3[0][0], 'conf': top3[0][1],
        })
        st.session_state.history = st.session_state.history[:5]

# ---------------------------------------------------------------------------
# Render the stored result. Because this reads from session state rather than
# from the button, it survives any later widget interaction. [Rules 3, 4, 6]
# ---------------------------------------------------------------------------
res = st.session_state.result
if res:
  with body:
      for note in res['notices']:
          st.warning(note)

      st.divider()
      st.subheader(t['result_header'])

      # Echo the submitted input so the user can confirm what was classified. [Rule 3]
      st.caption(
          f"{t['your_input']}: \"{res['desc']}\" / \"{res['act']}\" / {res['obj']}"
      )

      st.metric(t['predicted'], res['cause'])
      if res['cause'] == 'OTHER':
          st.info(t['other_note'])
      if res['top3'][0][1] < 0.90:
          st.info(t['low_conf'])

      # Top-3 candidates. Note: boosted trees trained on SMOTE-resampled data are
      # sharply over-confident (median top-1 probability around 0.99), so these
      # figures rank the candidates reliably but are not calibrated frequencies.
      st.write(f"**{t['top3']}**")
      for name, p in res['top3']:
          st.progress(min(max(float(p), 0.0), 1.0), text=f'{name} - {p*100:.1f}%')

      # Interpretability layer, clearly separated from the prediction itself.
      st.divider()
      st.markdown(f"**{t['explain_header']}**")
      st.caption(t['explain_note'])
      sym, act = res['symptom'], res['action']
      st.write(f"- {t['symptom']}: **{SYMPTOM_GLOSS.get(sym, {}).get(lang_choice, sym)}** (`{sym}`)")
      st.write(f"- {t['action']}: **{ACTION_GLOSS.get(act, {}).get(lang_choice, act)}** (`{act}`)")
      if sym == 'UNLABELED' or act == 'UNLABELED':
          st.caption(t['unrecognised'])

      # Explicit closure, telling the user the task is finished and what to do
      # next, rather than leaving the result hanging. [Rule 4]
      st.divider()
      st.success(t['done'])

# Session history, shown last so it never competes with the current result.
if st.session_state.history:
    with body.expander(f"{t['history']} ({len(st.session_state.history)})"):
        for i, h in enumerate(st.session_state.history, start=1):
            st.write(f"{i}. \"{h['desc']}\" / \"{h['act']}\" / {h['obj']} "
                     f"-> **{h['cause']}** ({h['conf']*100:.0f}%)")

