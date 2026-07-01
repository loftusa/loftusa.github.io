import NextAuth from "next-auth";
import GitHub from "next-auth/providers/github";

// The FastAPI backend base (same one the client uses). The signIn/jwt callback runs server-side.
const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "https://llm-resume-restless-thunder-9259.fly.dev";

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [
    GitHub({
      clientId: process.env.AUTH_GITHUB_ID,
      clientSecret: process.env.AUTH_GITHUB_SECRET,
    }),
  ],
  callbacks: {
    // On first sign-in, S2S-upsert the user into the FastAPI `users` table (gated by
    // INTERNAL_API_KEY) to get the canonical backend user_id + session_version. Persisted in
    // the NextAuth JWT so /api/api-token can mint API bearer tokens without re-hitting the DB.
    async jwt({ token, account, profile }) {
      if (account && profile) {
        try {
          const res = await fetch(`${API_BASE}/internal/users/upsert`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "X-Internal-Key": process.env.INTERNAL_API_KEY ?? "",
            },
            body: JSON.stringify({
              email: token.email,
              name: token.name,
              provider: account.provider,
              provider_sub: String((profile as { id?: string | number }).id ?? account.providerAccountId ?? ""),
            }),
          });
          if (res.ok) {
            const data = await res.json();
            (token as unknown as Record<string, unknown>).backendUserId = data.user_id;
            (token as unknown as Record<string, unknown>).sessionVersion = data.session_version;
          }
        } catch {
          // upsert is best-effort — a backend hiccup shouldn't block login
        }
      }
      return token;
    },
    async session({ session, token }) {
      (session as unknown as Record<string, unknown>).backendUserId = (token as unknown as Record<string, unknown>).backendUserId;
      (session as unknown as Record<string, unknown>).sessionVersion = (token as unknown as Record<string, unknown>).sessionVersion;
      return session;
    },
  },
});
