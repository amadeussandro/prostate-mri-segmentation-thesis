# Starting a new session

Paste the block below as the first message in a fresh context window. Nothing
else is needed — `HANDOFF.md` carries the state.

---

```
Baca HANDOFF.md dulu sebelum mengerjakan apa pun. Itu state dokumen proyek
skripsi ini — model yang sudah dibekukan, hasil final, keputusan yang tidak
boleh dibuka ulang, dan daftar pekerjaan terbuka.

Aturan kerja:
- Angka naskah tidak pernah diketik manual. Semuanya dimuat dari artefak hasil
  lewat scripts/manuscript_numbers.py. Kalau butuh angka, ambil dari situ.
- Naskah .docx digenerasi, bukan diedit tangan. Edit modul konten
  (scripts/manuscript_content_en.py / _id.py), lalu rebuild. Rebuild menimpa
  .docx.
- Pekerjaan CPU dikerjakan lokal, langsung kasih hasil. Hanya pekerjaan GPU
  yang dibuat jadi notebook Colab. Google Drive sudah ter-mount di
  G:\My Drive\THESIS_PROSTATE158, jadi data bukan lagi penghalang.
- Commit atas nama saya (lihat COMMIT_RULES.md), tanpa trailer AI. Jangan push
  tanpa izin saya.
- Tidak boleh ada penyebutan AI di mana pun dalam naskah, termasuk metadata
  dokumen.
- Setelah perubahan apa pun yang mengubah hasil, keputusan, atau daftar
  pekerjaan terbuka — perbarui HANDOFF.md di commit yang sama.

Setelah baca, konfirmasi ke saya: posisi sekarang di mana dan langkah
berikutnya apa.
```

---

## If the session is for the skripsi

Add this after the block above:

```
Fokus sesi ini: skripsi (Bab I-V, LaTeX), bukan jurnal. Jurnalnya sudah
selesai. Skripsi adalah dokumen berbeda dengan struktur berbeda.

Template LaTeX kampus ada di: <PATH>
```

Without the template, the structure has to be guessed and that means rework.
See HANDOFF.md §6.A.

## Keeping the handoff honest

`HANDOFF.md` is the only thing a new session reads. It is worth more than any
summary in a chat log, and it is worthless the moment it goes stale.

- Update it in the **same commit** as the change it describes.
- If a statement there contradicts an artifact, the artifact wins — fix the file.
- Delete finished work rather than marking it done twice; the file grew to 1,500
  lines once by accumulating patches, and parts of it ended up contradicting
  reality.
