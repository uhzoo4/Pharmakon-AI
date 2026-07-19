## UI Pro Max Search Results
**Domain:** gsap | **Query:** scroll reveal pin stagger split text headline
**Source:** motion.csv | **Found:** 4 results

### Result 1
- **Category:** Stagger List
- **Intensity Tier:** Complex
- **Trigger:** load or scroll
- **Duration:** 400-700ms
- **Easing:** expo.out
- **GSAP Snippet:** const split = new SplitText(headline, { type: 'chars' }); gsap.from(split.chars, { opacity: 0, y: 20, rotateX: -40, duration: 0.6, stagger: 0.015, ease: 'expo.out' });
- **Framework Notes:** SplitText is a GSAP Club/paid plugin; confirm license before shipping and provide a plain fade fallback if unavailable
- **Do:** Revert SplitText on unmount/cleanup (split.revert()) to restore original text nodes for accessibility tools
- **Don't:** Don't split-animate long paragraphs; reserve for short headlines (under ~8 words)
- **Performance Notes:** Splitting text creates one element per character; keep it to headline-length copy only for DOM size

### Result 2
- **Category:** Scroll Reveal
- **Intensity Tier:** Complex
- **Trigger:** scroll (continuous scrub)
- **Duration:** tied to scroll position
- **Easing:** none (scrub-driven)
- **GSAP Snippet:** gsap.timeline({ scrollTrigger: { trigger: section, start: 'top top', end: '+=150%', scrub: 1, pin: true } }).from('.headline', { opacity: 0, y: 40 }).to('.bg-layer', { yPercent: -20 }, '<');
- **Framework Notes:** Pinning needs the section to have deterministic height; recalc ScrollTrigger.refresh() after images/fonts load
- **Do:** Use scrub: true or a small number (0.5-1.5) instead of instant jumps so it feels tied to the scrollbar
- **Don't:** Don't pin more than 1-2 sections per page; excessive pinning fights native scroll feel and hurts mobile UX
- **Performance Notes:** Pinning forces layout reflow; test on mid-tier mobile devices, not just desktop

### Result 3
- **Category:** Scroll Reveal
- **Intensity Tier:** Subtle
- **Trigger:** scroll (viewport enter)
- **Duration:** 300-400ms
- **Easing:** power1.out
- **GSAP Snippet:** gsap.from(el, { opacity: 0, y: 12, duration: 0.35, ease: 'power1.out', scrollTrigger: { trigger: el, start: 'top 90%', toggleActions: 'play none none reverse' } });
- **Framework Notes:** Requires the ScrollTrigger plugin registered once via gsap.registerPlugin(ScrollTrigger)
- **Do:** Keep the y offset small (8-16px) so it reads as a fade, not a slide
- **Don't:** Don't reveal below-the-fold content needed for SEO/crawlers as invisible-by-default without a no-JS fallback
- **Performance Notes:** toggleActions 'play none none reverse' avoids re-triggering on every scroll direction change

### Result 4
- **Category:** Scroll Reveal
- **Intensity Tier:** Standard
- **Trigger:** scroll (viewport enter)
- **Duration:** 400-600ms
- **Easing:** power2.out
- **GSAP Snippet:** gsap.from(el.children, { opacity: 0, y: 24, duration: 0.5, stagger: 0.08, ease: 'power2.out', scrollTrigger: { trigger: el, start: 'top 85%' } });
- **Framework Notes:** In React use useGSAP(() => {...}, { scope: containerRef }) from @gsap/react to auto-cleanup on unmount
- **Do:** Scope the ScrollTrigger to the section container so it doesn't re-scan the whole page
- **Don't:** Don't stagger more than ~8 children; beyond that the last items feel laggy
- **Performance Notes:** Set scroller/markers: false in production; markers is dev-only

