# Dense frame review: abuse positives and high-scoring negatives

## Method and limits

Reviewed contact sheets made from up to 24 frames sampled uniformly across each selected clip. The positive set contains the eight lowest-scoring abuse examples from the source-group-held-out validation review; the negative set contains the eight highest abuse-scoring negatives from the official test diagnostics. The two priority clips below were then sampled at 1 frame per second across their entire annotated intervals. Their audio tracks were decoded and passed through the project's frozen YAMNet acoustic event tagger. This is still not continuous human playback or expert relabeling; YAMNet labels are weak model evidence and can be wrong, especially on film soundtracks. Conclusions are provisional and should not be used to change benchmark labels on their own.

Full-interval frame sheets: [City of God, seconds 1–20](full_context/city_of_god_horseplay_01-20s.jpg), [21–40](full_context/city_of_god_horseplay_21-40s.jpg), [41–54](full_context/city_of_god_horseplay_41-54s.jpg); [Black Hawk Down, seconds 1–20](full_context/black_hawk_down_normal_01-20s.jpg), [21–40](full_context/black_hawk_down_normal_21-40s.jpg), [41–49](full_context/black_hawk_down_normal_41-49s.jpg).

## Held-out abuse positives (lowest scores)

| Clip | Score | Visual screen |
|---|---:|---|
| Operation Red Sea, 00:33:45–00:34:20 | 0.087 | Frames appear to show a person being attacked or restrained in a crowd. Abuse label looks plausible. |
| City of God, 00:19:34–00:20:05 | 0.179 | Assault-related violence is visible, with possible overlap with shooting. Abuse label looks plausible but multi-category. |
| City of God, 00:59:55–01:02:35 | 0.320 | Injured/distressed child and armed-conflict context appear in the samples. The abuse interpretation depends on context; multi-label. |
| Taken, 00:36:58–00:37:40 | 0.404 | Physical beating appears in the latter sampled portion. Fighting and abuse labels can both fit. |
| City of God, 01:09:11–01:10:35 | 0.489 | Distress and men entering a room are visible, but the abuse event is not clear from stills alone. Needs full-context review. |
| Taken, 01:16:27–01:18:26 | 0.493 | Violent hostage/action scene; abuse, fighting, and shooting labels may overlap. |
| City of God, 00:27:11–00:28:05 | 0.548 | This 54-second cut changes scenes: the first ~19 seconds show an indoor episode with distress-like moments; roughly 20–48 seconds show boys/young men wrestling and interacting in a way that looks playful in sampled frames. YAMNet's strongest screaming tag is at clip-relative 13.5 seconds (0.74); its strongest crying/sobbing tag is at 44.5 seconds (0.46). These are model guesses, not confirmed sounds. The abuse label may refer to the earlier episode or broader context, so this is still the highest-priority positive for a human to watch and listen to. |
| Taken, 01:03:57–01:05:04 | 0.701 | Repeated beating/interrogation-like violence appears in the samples. Abuse label looks plausible. |

## Official test high-scoring negatives

| Clip | Predicted abuse score | Ground-truth label | Visual screen |
|---|---:|---|---|
| Mission: Impossible II, 00:01:29–00:01:44 | 0.809 | Fighting | Dark action/combat sequence; plausible adjacent-class confusion. |
| Taken 3, 01:19:13–01:19:30 | 0.784 | Fighting | Clear physical fight. Strong category-boundary example, not an obvious annotation error. |
| Taken 3, 01:04:50–01:05:05 | 0.766 | Fighting | Confrontation/fight context; frames include people approaching and a scuffle. Plausible adjacent-class confusion. |
| Kill Bill Vol. 1, 01:26:56–01:28:18 | 0.764 | Fighting | Stylized sword combat and subsequent violence. Plausible fighting/abuse semantic overlap. |
| Ip Man, 01:14:31–01:15:24 | 0.753 | Fighting | Multiple combat/assault frames. Plausible adjacent-class confusion. |
| Black Hawk Down, 01:36:44–01:37:33 | 0.750 | Normal | The full interval is a sustained military combat/war-zone sequence, including soldiers taking cover and moving through damaged streets. It is a strong visual hard negative under the list's normal label. YAMNet is dominated by speech (mean 0.305) and soundtrack music (mean 0.250); its maximum gunfire score is only 0.006 and explosion 0.011, so this tagger did not confirm audible gunfire. The label is worth auditing against the benchmark's annotation policy, but sparse evidence cannot establish a bad label. |
| The Bourne Identity, 00:19:53–00:21:48 | 0.720 | Fighting | Institutional/crowd scene with apparent disruption, but the sampled frames do not make an abuse event clear. Could reflect movie/action context; needs full playback to interpret. |
| City of God, 00:40:16–00:41:30 | 0.710 | Shooting | Violence/armed-conflict context appears in the samples. Plausible confusion with abuse, not an obvious label error. |

## Takeaway

Most reviewed false positives are fighting or shooting clips with visible violence, which supports class overlap as a major source of abuse errors. Human review by the project owner confirmed both labels for the abuse-detection task: the *City of God* clip is correctly labeled abuse; *Black Hawk Down* contains a gunfire exchange but no abuse and is correctly a negative for abuse. This distinction matters: a clip can depict a different incident category and still be a valid negative for the abuse class. YAMNet's low gunfire score did not capture the visually apparent exchange. The project's annotation file uses YouTube IDs and does not map directly to these movie-derived clip IDs; split-list labels confirm the provided tags. No relabeling is needed for this abuse-specific audit.
