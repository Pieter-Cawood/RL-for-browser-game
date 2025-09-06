"""Robust button clicking with Playwright for the app layer."""
from __future__ import annotations

from typing import Tuple

from playwright.sync_api import Locator, Page

from .config import ButtonSpec, ButtonType, ClickBehavior

class ButtonClicker:
    """Encapsulates resilient click behavior for intro buttons."""
    def __init__(self, page: Page, behavior: ClickBehavior):
        self.page = page
        self.cfg = behavior

    def _resolve(self, kind: ButtonType, value: str) -> Locator | None:
        if kind == "css":
            return self.page.locator(value)
        if kind == "text":
            return self.page.get_by_text(value, exact=True)
        if kind == "role":
            return self.page.get_by_role("button", name=value, exact=True)
        if kind == "xpath":
            return self.page.locator(f"xpath={value}")
        return None

    def _wait_enabled(self, loc: Locator, timeout: int = 3000) -> None:
        loc.wait_for(state="visible", timeout=timeout)
        loc.wait_for(state="attached", timeout=timeout)
        self.page.wait_for_timeout(self.cfg.stability_ms)

    def click(self, spec: ButtonSpec) -> bool:
        """
        Try to click a button defined by spec.
        If multiple matches exist, click the `instance`-th one (1-based).
        """
        t = spec.get("type", "css")  # type: ignore[assignment]
        v = spec.get("value", "")    # type: ignore[assignment]
        idx = spec.get("idx", 0) 

        strategies: Tuple[Tuple[ButtonType, str], ...] = tuple(
            [(t, v)] +
            ([("text", v)] if t not in ("text", "xpath") and v else []) +
            ([("role", v)] if t not in ("role", "xpath") and v else [])
        )  # type: ignore[list-item]

        last_err: Exception | None = None

        for _ in range(self.cfg.retries):
            for kind, val in strategies:
                try:
                    loc = self._resolve(kind, val)
                    if not loc:
                        continue

                    # If there aren't enough matches, skip this strategy early
                    try:
                        if hasattr(loc, "count") and loc.count() <= idx:
                            continue
                    except Exception:
                        # count() can throw before DOM settles; just continue to normal flow
                        pass

                    loc = loc.nth(idx)
                    loc.wait_for(state="visible", timeout=self.cfg.wait_visible_ms)
                    loc.scroll_into_view_if_needed(timeout=2000)
                    self._wait_enabled(loc, timeout=3000)
                    loc.click(timeout=3000)
                    self.page.wait_for_timeout(1000)
                    return True
                except Exception as e:
                    last_err = e

            # Fallback: focus + Enter
            try:
                loc = self._resolve(t, v) or self._resolve("role", v) or self._resolve("text", v)
                if loc:
                    try:
                        if hasattr(loc, "count") and loc.count() <= idx:
                            pass
                        else:
                            loc = loc.nth(idx)
                            loc.wait_for(state="visible", timeout=self.cfg.wait_visible_ms)
                            loc.focus(timeout=1500)
                            self.page.keyboard.press("Enter")
                            print(f"Pressed Enter on: {t}={v} (instance {idx})")
                            self.page.wait_for_timeout(200)
                            return True
                    except Exception as e:
                        last_err = e
            except Exception as e:
                last_err = e

            # Fallback: JS click (supports instance>1 for CSS/text/role; skip for xpath)
            if t != "xpath":
                try:
                    ok = self.page.evaluate(
                        """(spec, idx) => {
                            const t = spec.type, v = spec.value;
                            const i = Math.max(0, idx|0);
                            // CSS
                            if (t === 'css') {
                            const els = document.querySelectorAll(v);
                            if (els && els[i]) { els[i].click(); return true; }
                            return false;
                            }
                            // ROLE/TEXT: try buttons-like nodes and match text exactly
                            const pool = Array.from(document.querySelectorAll('button, [role="button"]'));
                            const matches = pool.filter(b => (b.textContent||'').trim() === v);
                            if (matches[i]) { matches[i].click(); return true; }
                            return false;
                        }""",
                        spec, idx
                    )
                    if ok:
                        print(f"DOM clicked via JS: {t}={v} (instance {idx})")
                        self.page.wait_for_timeout(200)
                        return True
                except Exception as e:
                    last_err = e

            # Fallback: mouse click center
            try:
                loc = self._resolve(t, v) or self._resolve("role", v) or self._resolve("text", v)
                if loc:
                    try:
                        if hasattr(loc, "count") and loc.count() <= idx:
                            pass
                        else:
                            loc = loc.nth(idx)
                            box = loc.bounding_box()
                            if box:
                                self.page.mouse.move(box["x"] + box["width"]/2, box["y"] + box["height"]/2)
                                self.page.mouse.down(); self.page.mouse.up()
                                print(f"Mouse clicked center: {t}={v} (instance {idx})")
                                self.page.wait_for_timeout(200)
                                return True
                    except Exception as e:
                        last_err = e
            except Exception as e:
                last_err = e

            self.page.mouse.wheel(0, 200)
            self.page.wait_for_timeout(self.cfg.backoff_ms)

        print(f"⚠️ Could not click {spec} (instance {idx}): {last_err}")
        return False


