---
layout: post
title: Ten Lessons from GitHub's First Year
---

h1. {{ page.title }}

p(meta). 29 Mar 2011 / 29 Dec 2008 - San Francisco

<b>NOTE: This post was written in late December of 2008, more than two years ago. It has stayed in my drafts folder since then, waiting for the last 2% to be written. Why I never published it is beyond my reckoning, but it serves as a great reminder of how I perceived the world back then. In the time since I wrote this we've grown from four people to twenty-six, settled into an office, installed a kegerator, and still never taken outside funding. In some ways, things have changed a great deal, but in the most important ways, things are still exactly the same. Realizing this puts a big smile on my face.</b>

The end of the year is a great time to sit down with a glass of your favorite beverage, dim the lights, snuggle up next to the fire and think about what you've learned over the past twelve months.

For me, 2008 was the year that I helped design, develop, and launch GitHub. Creating a new startup is an intense learning experience. Through screwups and triumphs, I have learned some valuable lessons this year. Here's a few of them.

h2. Start Early

When Chris and I started working on GitHub in late 2007, Git was largely unknown as a version control system. Sure, Linux kernel hackers had been using it since day one, but outside of that small microcosm, it was rare to come across a developer that was using it on a day-to-day basis. I was first introduced to Git by Dave Fayram, a good friend and former coworker during my days at Powerset. Dave is the quintessential early adopter and, as far as I can calculate, patient zero for the spread of Git adoption in the Ruby community and beyond.

Back then, the Git landscape was pretty barren. Git had only recently become  usable by normal people with the 1.5 release. As for Git hosting, there was really only "repo.or.cz":http://repo.or.cz/, which felt to me very limited, clumsy, and poorly designed. There were no commercial Git hosting options whatsoever. Despite this, people were starting to talk about Git at the Ruby meetups. About how awesome it was. But something was amiss. Git was supposed to be this amazing way to work on code in a distributed way, but what was the mechanism to securely share private code? Your only option was to setup user accounts on Unix machines and use that as an ad-hoc solution. Not ideal.

And so GitHub was born. But it was born into a world where there was no existing market for paid Git hosting. We would be _creating_ the market. I vividly remember telling people, "I don't expect GitHub to succeed right away. Git adoption will take a while, but we'll be ready when it happens." Neither Chris nor I were in any particular hurry for this to happen. I was working full time at Powerset, and he was making good money as a Rails consultant. By choosing to build early on top of a nascent technology, we were able to construct a startup with basically no overhead, no competition, and in our free time.

h2. Adapt to Your Customers

Here's a seemingly paradoxical piece of advice for you: Listen to your customers, but don't let them tell you what to do. Let me explain. Consider a feature request such as "GitHub should let me FTP up a documentation site for my project." What this customer is really trying to say is "I want a simple way to publish content related to my project," but they're used to what's already out there, and so they pose the request in terms that are familiar to them. We could have implemented some horrible FTP based solution as requested, but we looked deeper into the underlying question and now we allow you to publish content by simply pushing a Git repository to your account. This meets requirements of both functionality _and_ elegance.

Another company that understands this concept at a fundamental level is Apple. I'm sure plenty of people asked Apple to make a phone but Steve Jobs and his posse looked beneath the request and figured out what people really wanted: a nice looking, simple to use, and easy to sync mobile device that kicked some serious ass. And that's the secret. Don't give your customers what they ask for; give them what they _want_.

h2. Have Fun

I went to college at a little school in California called Harvey Mudd. Yeah, I know you haven't heard of it, but if you remember those US News & World Report "Best Colleges" books that you obsessed over in highschool (ok, maybe you didn't, but I did), Harvey Mudd was generally ranked as the engineering school with the greatest number of hours of homework per night. Yes, more than MIT, and yes, more than Caltech. It turned out to be true, as far as I can tell. I have fond memories of freaking out about ridiculously complex spring/mass/damper systems and figuring the magnetic flux of a wire wrapped around a cylinder in a double helix. We studied hard--very hard. But we played hard too. It was the only thing that could possibly keep us sane.

Working on a startup is like that. It feels a bit like college. You're working on insanely hard projects, but you're doing it with your best friends in the world and you're having a great time (usually). In both environments, you have to goof off a lot in order to balance things out. Burnout is a real and dangerous phenomenon. Fostering a playful and creative environment is critical to maintaining both your personal health, and the health (and idea output) of the company.

h2. Pay attention to Twitter

I've found Twitter to be an extremely valuable resource for instant feedback. If the site is slow for some reason, Twitter will tell me so. If the site is unreachable for people in a certain country (I'm looking at you China), I'll find out via Twitter. If that new feature we just released is really awesome, I'll get a nice ego boost by scanning the Twitter search.

People have a tendency to turn to Twitter to bitch about all the little bugs they see on your website, usually appended with the very tiresome "FAIL". These are irksome to read, but added together are worth noticing. Often times these innocent tweets will inform a decision about whether an esoteric bug is worth adding to the short list. We also created a GitHub account on Twitter that our support guy uses to respond to negative tweets. Offering this level of customer service almost always turns a disgruntled customer into a happy one.

If you have an iPhone, I heartily recommend the "Summizer":http://fanzter.com/products/download/summizer app from Fanzter, Inc. It makes searching, viewing, and responding to tweets a cinch.

h2. Deploy at Will!

At the first RailsConf I had the pleasure of hearing Martin Fowler deliver an amazing keynote. He made some apt metaphors regarding agile development that I will now paraphrase and mangle.

Imagine you're tasked with building a computer controlled gun that can accurately hit a target about 50 meters distant. That is the only requirement. One way to do this is to build a complex machine that measures every possible variable (wind, elevation, temperature, etc.) before the shot and then takes aim and shoots. Another approach is to build a simple machine that fires rapidly and can detect where each shot hits. It then uses this information to adjust the aim of the next shot, quickly homing in on the target a little at a time.

The difference between these two approaches is to realize that bullets are cheap. By the time the former group has perfected their wind detection instrument, you'll have finished your simple weapon and already hit the target.

In the world of web development, the target is your ideal offering, the bullets are your site deploys, and your customers provide the feedback mechanism. The first year of a web offering is a magical one. Your customers are most likely early adopters and love to see new features roll out every few weeks. If this results in a little bit of downtime, they'll easily forgive you, as long as those features are sweet. In the early days of GitHub, we'd deploy up to ten times in one afternoon, always inching closer to that target.

Make good use of that first year, because once the big important customers start rolling in, you have to be a lot more careful about hitting one of them with a stray bullet. Later in the game, downtime and botched deploys are money lost and you have to rely more on building instruments to predict where you should aim.

h2. You Don't Need an Office

All four fulltime GitHub employees work in the San Francisco area, and yet we have no office. But we're not totally virtual either. In fact, a couple times a week you'll find us at a cafe in North Beach, huddled around a square table that was made by nailing 2x4s to an ancient fold-out bulletin board. It's no Google campus, but the rent is a hell of a lot cheaper and the drinks are just as good!

This is not to say that we haven't looked at a few places to call home. Hell, we almost leased an old bar. But in the end there's no hurry to settle down. We're going to wait until we find the perfect office. Until then, we can invest the savings back into the company, or into our pockets. For now, I like my couch and the cafe just fine.

Of course, none of this would be possible without 37signals' "Campfire":http://www.campfirenow.com/ web-based chat and the very-difficult-to-find-but-totally-amazing "Propane":http://productblog.37signals.com/products/2008/10/propane-takes-c.html OSX desktop app container that doubles the awesome. Both highly recommended.

h2. Hire Through Open Source

Beyond the three cofounders of GitHub, we've hired one full time developer (Scott Chacon) and one part time support specialist (Tekkub).

We hired Tekkub because he was one of the earliest GitHub users and actively maintains more than 75 projects (WoW addons mostly) on GitHub and was very active in sending us feedback in the early days. He would even help people out in the IRC channel, simply because he enjoyed doing so.

I met Scott at one of the San Francisco Ruby meetups where he was presenting on one of his myriad Git-centric projects. Scott had been working with Git long before anyone else in the room. He was also working on a pure Ruby implementation of Git at the same time I was working on my fork/exec based Git bindings. It was clear to me then that depending on how things went down, he could become either a powerful ally or a dangerous foe. Luckily, we all went drinking afterwards and we became friends. Not long after, Scott started consulting for us and wrote the entire backend for what you now know of as "Gist":http://gist.github.com/. We knew then that we would do whatever it took to hire Scott full time. There would be no need for an interview or references. We already knew everything we needed to know in order to make him an offer without the slightest reservation.

The lesson here is that it's far easier and less risky to hire based on relevant past performance than it is to hire based on projected future performance. There's a corollary that also comes into play: if you're looking to work for a startup (or anyone for that matter), contribute to the community that surrounds it. Use your time and your code to prove that you're the best one for the job.

h2. Trust your Team

There's nothing I hate more than micromanagers. When I was doing graphic design consulting 5 years ago I had a client that was very near the Platonic form of a micromanager. He insisted that I travel to his office where I would sit in the back room at an old Mac and design labels and catalogs and touch up photographs of swimwear models (that part was not so bad!). While I did these tasks he would hover over me and bark instructions. "Too red! Can you make that text smaller? Get rid of those blemishes right there!" It drove me absolutely batty.

This client could have just as easily given me the task at the beginning of the day, gone and run the business, and come back in 6 hours whereupon I would have created better designs twice as fast as if he were treating me like a robot that converted his speech into Photoshop manipulations. By treating me this way, he was marginalizing my design skills and wasting both money and talent.

Micromanagement is symptomatic of a lack of trust. The remedy for this ailment is to hire experts and then trust their judgment. In a startup, you can drastically reduce momentum by applying micromanagement, or you can boost momentum by giving trust. It's pretty amazing what can happen when a group of talented people who trust each other get together and decide to make something awesome.

h2. You Don't Need Venture Capital

A lot has been written recently about how the venture capital world is changing. I don't pretend to be an expert on the subject, but I've learned enough to say that a web startup like ours doesn't need any outside money to succeed. I know this because we haven't taken a single dime from investors. We bootstrapped the company on a few thousand dollars and became profitable the day we opened to the public and started charging for subscriptions.

In the end, every startup is different, and the only person that can decide if outside money makes sense is you. There are a million things that could drive you to seek and accept investment, but you should make sure that doing so is in your best interest, because it's quite possible that you don't _need_ to do so. One of the reasons I left my last job was so that I could say "the buck stops here." If we'd taken money, I would no longer be able to say that.

h2. Open Source Whatever You Can

In order for GitHub to talk to Git repositories, I created the first ever Ruby Git bindings. Eventually, this library become quite complete and we were faced with a choice: Do we open source it or keep it to ourselves? Both approaches have benefits and drawbacks. Keeping it private means that the hurdle for competing Ruby-based Git hosting sites would be higher, giving us an advantage. But open sourcing it would mean that

<b>NOTE: This is where the post ended and remained frozen in carbonite until today. I intend to write a follow up post on our open source philosophy at GitHub in the near future. I'm sure the suspense is killing you!</b>

"Discuss this post on Hacker News":http://news.ycombinator.com/item?id=2384320
