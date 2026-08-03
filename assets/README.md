# Icons

Every file here is generated from the Railway Academy logo by
`make_icons.py`, which pulls the 512px original the site uses as its own
favicon:

<https://railwayacademy.org/wp-content/uploads/2025/11/FAVICON.png>

The logo is Railway Academy's, used here for their own project. Don't
edit the PNGs by hand, re-run the script instead:

```
python assets/make_icons.py
```

| File | Used for |
| --- | --- |
| `logo.png` | 512px master, and the sidebar mark via `st.logo` |
| `icon-16/32/48/64/128/192/256/512.png` | browser tab, launcher and PWA sizes |
| `apple-touch-icon.png` | 180px, iOS home screen |
| `favicon.ico` | multi-size 16-256, for a non-Streamlit host |

`app.py` uses `icon-192.png` as the Streamlit `page_icon` and
`logo.png` / `icon-64.png` for `st.logo`. The `.ico` and
`apple-touch-icon.png` aren't wired in - Streamlit sets the tab icon
itself - they're here for whatever the app ends up being deployed behind.
